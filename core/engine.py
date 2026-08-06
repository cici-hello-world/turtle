"""海龟策略回测引擎

支持多种模式:
- "long":                    不加仓做多
- "long_more":               加仓做多（金字塔）
- "short_more":              加仓做空（金字塔）
- "both":                    多空双向
- "anti_martingale":         反马丁格尔（MA信号 + 跨交易仓位调整）
- "position_management_1":   仓位管理策略1（网格：参考价±N%触发买卖）
- "position_management_2":   仓位管理策略2（Swing恢复：反弹卖出、回落买回，固定手数）
"""

import copy
import math
import numpy as np
import pandas as pd


def run_backtest(df: pd.DataFrame, config) -> tuple[list[dict], list[float], list]:
    """统一回测入口，返回 (trades, equity_curve, equity_dates)

    equity_dates 与 equity_curve 一一对应，均为日频（每个交易日一个点）。
    """
    if config.mode == "long":
        return _run_long_only(df, config)
    elif config.mode == "long_more":
        return _run_long_with_pyramid(df, config)
    elif config.mode == "short_more":
        return _run_short_with_pyramid(df, config)
    elif config.mode == "both":
        return _run_long_short(df, config)
    elif config.mode == "anti_martingale":
        return _run_anti_martingale(df, config)
    elif config.mode == "position_management_1":
        return _run_position_management_1(df, config)
    elif config.mode == "position_management_2":
        return _run_position_management_2(df, config)
    else:
        raise ValueError(f"未知的策略模式: {config.mode}")


# ==================== 共享工具函数 ====================

def _ema_trend_ok(row, config, direction: str = "long") -> bool:
    """EMA 趋势过滤器。direction="long" 要求快线在慢线之上。"""
    fast = config.ema_fast
    slow = config.ema_slow
    if fast <= 0 or slow <= 0:
        return True  # 参数为 0 表示禁用过滤器
    fast_val = row.get("EMA_Fast")
    slow_val = row.get("EMA_Slow")
    if pd.isna(fast_val) or pd.isna(slow_val):
        return True  # 数据不足时不拦截
    if direction == "long":
        return fast_val > slow_val
    else:
        return fast_val < slow_val

def _dmi_trend_ok(row, config, direction: str = "long") -> bool:
    """dmi 趋势过滤器。"""
    plus_di = row.get("+DI", np.nan)
    minus_di = row.get("-DI", np.nan)
    adx = row.get("ADX14", np.nan)
    adxr = row.get("ADXR", np.nan)
    adx_up = row.get("ADX_UP", np.nan)
    adx_down = row.get("ADX_DOWN", np.nan)
    acua = row.get("ADX_CROSS_UP_ADXR", np.nan)
    acda = row.get("ADX_CROSS_DOWN_ADXR", np.nan)
    bonus = adx_up and acua
    trend_turn_around = adx_down and not acda #趋势调头但未衰竭
    dmi_ok = (not pd.isna(adx) and not pd.isna(plus_di)
              and not pd.isna(minus_di) and not pd.isna(adxr)) and adx > 35 #强趋势才能入场
    if not config.user_dmi:
        return True
    if dmi_ok:
        if direction == "long":
            is_bull_trend = (plus_di - minus_di)>5 #多头趋势
        else:
            is_bull_trend = plus_di < minus_di #空头趋势
        trend_accelerating = bonus or trend_turn_around  # 强趋势
    else:
        is_bull_trend = False
        trend_accelerating = False
    return is_bull_trend and trend_accelerating

def _dmi_trend_area(row, config, direction: str = "long",
                     baseline: dict | None = None) -> str:
    """dmi 趋势过滤器（个股自适应版）。

    三区定义：
      A区 — 趋势确认：多头成立 + ADX_SLOPE_5 > slope_sep + 趋势结构确认
            · 标准：ADX>阈值 且 ADX>ADXR
            · DI强势：|+DI - -DI| > di_gap 且 ADX>阈值
      C区 — 趋势酝酿：不满足A + ADX_SLOPE_5 > slope_sep
      D区 — 无趋势震荡：ADX_SLOPE_5 ≤ slope_sep

    阈值来源（优先级）：
      1. baseline 参数（来自 df.attrs["dmi_baseline"]，compute_indicators 自动计算）
      2. 硬编码默认值（基于 09992.HK 1343天实证，与归一化 DMI 指标跨个股通用）

    direction 参数：
      "long"  → 多头需 +DI - -DI > di_dir
      "short" → 多头需 -DI - +DI > di_dir
    """
    # ── 读取阈值：baseline 优先，否则默认 ──
    bl = baseline or {}
    slope_sep = bl.get("slope_sep", -0.05)
    di_gap = bl.get("di_gap", 10.0)
    di_dir = bl.get("di_dir", 5.0)

    plus_di = row.get("+DI", np.nan)
    minus_di = row.get("-DI", np.nan)
    adx = row.get("ADX14", np.nan)
    adxr = row.get("ADXR", np.nan)
    adx_slope = row.get("ADX_SLOPE_5", 0.0)
    if pd.isna(adx_slope):
        adx_slope = 0.0

    dmi_ok = (not pd.isna(adx) and not pd.isna(plus_di)
              and not pd.isna(minus_di) and not pd.isna(adxr))
    if not config.user_dmi:
        return None
    if dmi_ok:
        # —— 方向判断 ——
        if direction == "long":
            is_bull_trend = (plus_di - minus_di) > di_dir
        else:
            is_bull_trend = (minus_di - plus_di) > di_dir
        threshold = config.pm_adx_threshold

        # —— 趋势结构确认（两条路径） ——
        # 标准：ADX>阈值 且 ADX>ADXR
        trend_standard = adx > threshold and adx > adxr
        # DI强势：|+DI - -DI| > di_gap，ADX>阈值
        trend_di_strong = abs(plus_di - minus_di) > di_gap and adx > threshold

        # A区：趋势确认
        if is_bull_trend and adx_slope > slope_sep and (trend_standard or trend_di_strong):
            _trend_area = "A"
        elif adx_slope > slope_sep:
            _trend_area = "C"
        else:
            _trend_area = "D"
    else:
        return None
    return _trend_area


# ── 自适应网格的个股经验常量（基于20日forward收益 + 波动率实证）──
# zone_score: A/D/C 区 → blend 映射值，越高越积极
# vol_base: ADX=30 时的基准 ATR/Close
# vol_adx_slope: ADX 每偏离30多少点 → vol_threshold 调 0.01（越大=越不敏感）
# vol_div: vol_adj 归一化除数（≈该股票 vol 的 std）
# 默认值来自 09992.HK 泡泡玛特，其他股票按需覆盖
_STOCK_GRID_CONSTANTS = {
    "09992.HK": dict(  # 泡泡玛特
        zone_score={"A": 1.0, "D": 0.65, "C": 0.30},
        vol_base=0.05, vol_adx_slope=600.0, vol_div=0.08,
    ),
    "03690.HK": dict(  # 美团 — D区+1.34% C区-0.32%，D远弱于泡泡玛特
        zone_score={"A": 1.0, "D": 0.40, "C": 0.25},
        vol_base=0.047, vol_adx_slope=1200.0, vol_div=0.028,
    ),
    "06181.HK": dict(  # 老铺 — A区+29% D区+7.7% C区-0.97%，A极强但D也不弱
        zone_score={"A": 1.0, "D": 0.45, "C": 0.25},
        vol_base=0.061, vol_adx_slope=1200.0, vol_div=0.025,
    ),
    "01810.HK": dict(  # 小米 — A区+3.4% D区+1.8% C区-0.82%，与泡泡玛特格局接近
        zone_score={"A": 1.0, "D": 0.65, "C": 0.25},
        vol_base=0.040, vol_adx_slope=1200.0, vol_div=0.020,
    ),
    "02015.HK": dict(  # 理想 — A区-1.65% D区+0.68% C区-0.33%，DMI区制预测力弱
        zone_score={"A": 0.30, "D": 0.45, "C": 0.20},
        vol_base=0.055, vol_adx_slope=800.0, vol_div=0.035,
    ),
}


def _get_grid_constants(ts_code: str) -> dict:
    """获取个股自适应网格常量，未配置的股票返回默认值（09992.HK）。"""
    return _STOCK_GRID_CONSTANTS.get(ts_code)


def _get_best_zone(ts_code: str) -> str:
    """获取个股 zone_score 中权重最高的区域，用于 DMI 阻塞判断。

    未配置的股票默认返回 "D"（无趋势震荡区）。
    """
    gc = _get_grid_constants(ts_code)
    if gc and gc.get("zone_score"):
        return max(gc["zone_score"], key=gc["zone_score"].get)
    return "D"


def _calc_effective_grid(row, config, baseline: dict | None = None) -> tuple[float, float, float]:
    """根据今日 DMI 状态与波动率计算有效网格 (eff_down, eff_up, eff_reduce)。

    由 _run_position_management_1 和 signal_extractor 共同调用，
    保证信号计算与回测引擎完全一致。

    核心改进（不对称网格）：
      趋势越强 → eff_down↓（回调敢加仓）  eff_up↑（不轻易卖）
      趋势越弱 → eff_down↑（不轻易抄底）  eff_up↓（快进快出）

    双方向区制（long×short 交叉）：
      分值基于20日forward收益实证：B(+6.37%)>A(+4.69%)>D(+3.25%)>C(+0.56%)
      zone_blend = long_score - short_score → long强+short弱=积极，反之防守
      strength 三因子（ADX位置/DI差值/ADX斜率）在 zone_blend 基础上微调
      DMI 不可用 → 返回原始网格
    """
    k = config.pm_agg_k
    gd = config.pm_grid_down
    gu = config.pm_grid_up
    rp = config.pm_reduce_pct

    # 激进高减仓比例（k<1 时放大卖出比例）
    rp_high = min(round(rp + (1 - k) * 0.5, 2), 0.95)

    if not config.pm_adaptive_grid:
        return gd, gu, rp

    area_long = _dmi_trend_area(row, config, "long", baseline)
    area_short = _dmi_trend_area(row, config, "short", baseline)

    if area_long is None or area_short is None:
        return gd, gu, rp                        # DMI 不可用 → 默认原始网格

    # ── 趋势强度三因子 (0~1) ──
    adx = row.get("ADX14", 20)
    plus_di = row.get("+DI", 0)
    minus_di = row.get("-DI", 0)
    threshold = config.pm_adx_threshold

    # 1) ADX 强度分：ADX 在 [threshold×0.5, threshold×1.5] 区间线性映射
    adx_score = (min(max(adx, threshold * 0.5), threshold * 1.5) - threshold * 0.5) / threshold
    adx_score = min(max(adx_score, 0.0), 1.0)

    # 2) DI 方向强度分：|+DI - -DI| 越大方向越明确
    di_score = min(abs(plus_di - minus_di) / 25.0, 1.0)

    # 3) ADX 斜率分（连续变量，替代二元 ADX_UP）
    adx_slope = row.get("ADX_SLOPE_5", 0.0)
    if pd.isna(adx_slope):
        adx_slope = 0.0
    adx_dir_score = min(max(0.3 + adx_slope * 5.0, 0.0), 1.0)

    strength = 0.40 * adx_score + 0.40 * di_score + 0.20 * adx_dir_score

    # ── 波动率微调（高波动略放宽防假突破，基于ADX自校准阈值）──
    atr = row.get("ATR", 0)
    close = row.get("Close", 0)
    gc = _get_grid_constants(config.ts_code)
    if atr > 0 and close > 0:
        vol = atr / close
        vol_threshold = gc["vol_base"] + (adx - 30.0) / gc["vol_adx_slope"]
        vol_adj = (vol - vol_threshold) / gc["vol_div"]
        vol_adj = min(max(vol_adj, -0.10), 0.10)
    else:
        vol_adj = 0.0

    # ── 双方向区制映射（分值基于20日forward收益实证）──
    zone_score = gc["zone_score"]
    long_score = zone_score.get(area_long, zone_score["C"])
    short_score = zone_score.get(area_short, zone_score["C"])
    # zone_blend: short打5折避免同区抵消，LD/SD(0.32) > LC/SC(0.15)有区分
    zone_blend = max(long_score - short_score * 0.5, 0.0)

    # strength 在 zone_blend 基础上 ±0.1 微调
    blend = zone_blend + (strength - 0.5) * 0.15
    blend = min(max(blend + vol_adj, 0.3), 0.7)

    # ── 不对称插值 ──
    # gd: blend↑ → gd↓（从 gd 插值到 gd*k，强趋势敢买回调）
    # gu: blend↑ → gu↑（从 gu*k 插值到 gu，强趋势不轻易卖）
    # rp: blend↑ → rp↓（从 rp_high 插值到 rp，强趋势卖得少）
    eff_down = round(gd * k + gd * (1 - k) * (1 - blend), 3)
    eff_up = round(gu * k + gu * (1 - k) * blend, 3)
    eff_reduce = round(rp_high + (rp - rp_high) * blend, 3)

    return eff_down, eff_up, eff_reduce


def _calc_shares(capital: float, risk_per_trade: float, stop_dist: float,
                 per_shares: int) -> int:
    """计算入场股数（手数），基于风险资金 / 止损距离"""
    return int(capital * risk_per_trade / (stop_dist * per_shares))


def _can_afford(cost: float, capital: float, ratio: float = 0.8) -> bool:
    """检查资金是否足够"""
    return cost <= capital * ratio


def _collect_dividend(row, shares: int, ps: int, capital: float):
    """除权日发放现金分红，返回 (dividend_income, new_capital)"""
    div = row.get("Cash_Dividend", 0.0) or 0.0
    if div > 0:
        income = div * shares * ps
        return income, capital + income
    return 0.0, capital


def _prep_entry(row, capital, config, per_shares):
    """ATR 校验 + 计算止损距离和股数。
    返回 (stop_dist, shares)；无效时返回 (0, 0)，调用方应 continue。
    """
    if pd.isna(row["ATR"]) or row["ATR"] == 0:
        return 0, 0
    atr = row["ATR"]
    stop_dist = config.n_value * atr
    shares = _calc_shares(capital, config.risk_per_trade, stop_dist, per_shares)
    if shares <= 0:
        return 0, 0
    return stop_dist, shares


def _build_trade(entry_date, exit_date, pos_type, entry_price, exit_price,
                 shares, cost, pnl, capital, reason, *,
                 stop_dist=0.0, more_info=None):
    """构建统一格式的交易记录"""
    record = {
        "Entry Date": entry_date,
        "Exit Date": exit_date,
        "Type": pos_type,
        "Entry Price": entry_price,
        "Exit Price": exit_price,
        "Shares": shares,
        "Cost_¥": cost,
        "PnL_¥": pnl,
        "PnL_%": pnl / cost if cost else 0.0,
        "Capital_After": capital,
        "Reason": reason,
    }
    if stop_dist:
        record["2ATR"] = stop_dist
    if more_info is not None:
        record["More Info"] = more_info
    return record


# ==================== 通道信号注册表（扩展新通道只需在此添加一项） ====================

_CHANNEL = {
    "donchian": {
        "long_entry":  lambda row, cfg: row["Close"] > row["HH_Entry"],
        "short_entry": lambda row, cfg: row["Close"] < row["LL_Entry"],
        "long_exit":   lambda row, cfg: row["Close"] < row["LL_Exit"],
        "short_exit":  lambda row, cfg: row["Close"] > row["HL_Exit"],
        "long_exit_reason":  "Trend Reversal (10d Low)",
        "short_exit_reason": "Trend Reversal (10d High)",
    },
    "bollinger": {
        "long_entry":  lambda row, cfg: row["Close"] > row["BB_Upper"],
        "short_entry": lambda row, cfg: row["Close"] < row["BB_Lower"],
        "long_exit":   lambda row, cfg: row["Close"] < row["BB_Mid"],
        "short_exit":  lambda row, cfg: row["Close"] > row["BB_Mid"],
        "long_exit_reason":  "Trend Reversal (BB Mid)",
        "short_exit_reason": "Trend Reversal (BB Mid)",
    },
}


def _get_channel(config) -> dict:
    """获取当前通道的信号定义"""
    return _CHANNEL[config.channel_type]


def _long_entry_triggered(row, config) -> bool:
    """做多入场信号"""
    return _get_channel(config)["long_entry"](row, config)


def _short_entry_triggered(row, config) -> bool:
    """做空入场信号"""
    return _get_channel(config)["short_entry"](row, config)


def _check_long_exit(row, stop_loss_price, config):
    """检查做多出场条件。返回 (exit_price, reason)；无信号返回 (None, '')。"""
    # 止损（所有通道共用）
    if row["Close"] < stop_loss_price:
        return row["Close"], "Hard Stop Loss (2×ATR)"
    # 通道特定的趋势出场
    ch = _get_channel(config)
    if ch["long_exit"](row, config):
        return row["Close"], ch["long_exit_reason"]
    return None, ""

def _check_short_exit(row, stop_loss_price, config):
    """检查做多出场条件。返回 (exit_price, reason)；无信号返回 (None, '')。"""
    # 止损（所有通道共用）
    if row["Close"] > stop_loss_price:
        return row["Close"], "Hard Stop Loss (2×ATR)"
    ch = _get_channel(config)
    if ch["short_exit"](row, config):
        return row["Close"], ch["short_exit_reason"]
    return None, ""


def _check_both_exit(row, pos, stop_loss_price, config):
    """检查双向出场条件。返回 (exit_price, reason)；无信号返回 (None, '')。"""
    if pos == "long":
        return _check_long_exit(row, stop_loss_price, config)
    else:  # short
        return _check_short_exit(row, stop_loss_price, config)


# ==================== Anti-Martingale 信号函数（独立于通道注册表） ====================

def _am_entry_triggered(row, config) -> bool:
    """Anti-Martingale 入场信号：Close > MA(period)"""
    ma = row.get("AM_MA")
    if pd.isna(ma):
        return False
    return row["Close"] > ma


def _am_exit_triggered(row, config) -> bool:
    """Anti-Martingale 出场信号：Close < MA(period)"""
    ma = row.get("AM_MA")
    if pd.isna(ma):
        return False
    return row["Close"] < ma


# ==================== 输出格式化 ====================

def _print_header(label, initial_capital, risk_percent):
    print(f"\n🔍 开始运行 {label} (带动态仓位管理)...")
    print(f"   初始资金：{initial_capital:,.2f} | 单笔风险：{risk_percent:.1f}%")


def _print_entry(label, price, shares, cost, capital, stop_loss):
    print(f"  🟢 [{label}] 做多入场：{price:.2f} | "
          f"股数：{shares} | 投入：{cost:.2f} | "
          f"剩余资金：{capital:.2f} | 止损：{stop_loss:.2f}")


def _print_short_entry(label,ts_code, price, shares, income, capital, stop_loss):
    print(f"  🟢 [{label}] [{ts_code}] 做空入场：{price:.2f} | "
          f"股数：{shares} | 收入：{income:.2f} | "
          f"剩余资金：{capital:.2f} | 止损：{stop_loss:.2f}")


def _print_pyramid_add(label, price, shares, cost, capital, stop_loss):
    print(f"  🟢 [{label}] 做多加仓：{price:.2f} | "
          f"股数：{shares} | 投入：{cost:.2f} | "
          f"剩余资金：{capital:.2f} | 止损上移至：{stop_loss:.2f}")


def _print_exit(label, pnl, capital, reason):
    status = "🟢 盈利" if pnl > 0 else "🔴 亏损"
    print(f"  {status} [{label}] 平仓：{reason} | "
          f"盈亏：{pnl:,.2f}¥ | 剩余资金：{capital:,.2f}")


def _print_force_close(label, pnl, capital):
    print(f"  ⚠️ [{label}] 强制平仓：{pnl:,.2f}¥ | 最终资金：{capital:,.2f}")


# ==================== 不加仓做多 ====================

def _run_long_only(df: pd.DataFrame, config) -> tuple[list[dict], list[float], list]:
    label = config.channel_type
    ps = config.per_shares

    trades = []
    capital = config.initial_capital

    start_index = config.entry_period + 1
    start_date = df.iloc[start_index]["Date"]
    equity_curve = [config.initial_capital]
    equity_dates = [start_date]

    pos = None
    entry_price = 0.0
    stop_loss_price = 0.0
    stop_dist = 0.0
    current_shares = 0
    cumulative_div = 0.0
    entry_date = None

    for i in range(start_index, len(df)):
        row = df.iloc[i]
        current_date = row["Date"]
        close = float(row["Close"])

        def _snap():
            equity_curve.append(capital + current_shares * close * ps)
            equity_dates.append(current_date)

        if pos is None:
            stop_dist, shares = _prep_entry(row, capital, config, ps)
            if shares <= 0:
                _snap()
                continue

            if _long_entry_triggered(row, config) and _ema_trend_ok(row, config):
                cost = row["Close"] * shares * ps
                if not _can_afford(cost, capital):
                    _snap()
                    continue
                pos = "long"
                entry_price = row["Close"]
                stop_loss_price = entry_price - stop_dist
                current_shares = shares
                cumulative_div = 0.0
                entry_date = current_date
                capital -= cost

        else:
            div_income, capital = _collect_dividend(row, current_shares, ps, capital)
            cumulative_div += div_income

            exit_price, exit_reason = _check_long_exit(row, stop_loss_price, config)
            if exit_reason:
                cost_basis = entry_price * current_shares * ps
                pnl = (exit_price - entry_price) * current_shares * ps + cumulative_div
                capital += exit_price * current_shares * ps

                trades.append(_build_trade(
                    entry_date, current_date, "Long",
                    entry_price, exit_price, current_shares,
                    cost_basis, pnl, capital, exit_reason,
                    stop_dist=stop_dist,
                ))
                pos = None
                current_shares = 0
                cumulative_div = 0.0
                entry_price = 0.0
                stop_loss_price = 0.0
                entry_date = None

        _snap()

    # 强制平仓
    if pos == "long":
        last_price = df.iloc[-1]["Close"]
        last_date = df.iloc[-1]["Date"]
        cost_basis = entry_price * current_shares * ps
        pnl = (last_price - entry_price) * current_shares * ps + cumulative_div
        capital += last_price * current_shares * ps

        trades.append(_build_trade(
            entry_date, last_date, "Long",
            entry_price, last_price, current_shares,
            cost_basis, pnl, capital, "强制平仓",
            stop_dist=stop_dist,
        ))
        # 强制平仓后补一个权益快照
        equity_curve.append(capital + current_shares * last_price * ps)
        equity_dates.append(last_date)

    return trades, equity_curve, equity_dates


# ==================== 加仓做多（金字塔） ====================

def _run_long_with_pyramid(df: pd.DataFrame, config,only_one=False,current_lots :int =0) -> tuple[list[dict], list[float], list]:
    label = config.channel_type
    ps = config.per_shares

    trades = []
    capital = config.initial_capital
    start_index = config.entry_period + 1
    if config.don_start_date != "":
        start_index = df.index.get_loc(df[df["Date"] == config.don_start_date].index[0])

    pos = None
    entry_price = 0.0
    entry_atr = 0.0
    stop_loss_price = 0.0
    stop_dist = 0.0
    current_shares = 0
    base_shares = 0
    max_shares = 0
    total_cost = 0.0
    cumulative_div = 0.0
    entry_date = None
    more_info = []
    equity_curve = []
    equity_dates = []
    for i in range(start_index, len(df)):
        if config.mode == "position_management_1" and i==start_index+1 and pos is None:
            _snap()
            return trades, equity_curve, equity_dates
        row = df.iloc[i]
        current_date = row["Date"]
        close = float(row["Close"])

        def _snap():
            equity_curve.append(capital + (current_shares + current_lots) * close * ps)
            equity_dates.append(current_date)

        if pos is None:
            if pd.isna(row["ATR"]) or row["ATR"] == 0:
                _snap()
                continue
            atr = row["ATR"]
            stop_dist = config.n_value * atr
            # Turtle 1 Unit = 1% Account ÷ (N × Dollars per Point)
            base_shares = int(0.01 * capital / (atr * ps))
            if base_shares <= 0:
                _snap()
                continue
            shares = base_shares
            skip_offset = 1 if config.skip_first_add else 0
            max_shares = shares * (config.max_pyramid - skip_offset)

            if (_long_entry_triggered(row, config)
                    # and _dmi_trend_ok(row, config)
                    and _ema_trend_ok(row, config)):
                cost = row["Close"] * shares * ps
                trade_cost = _pm_trade_cost(cost,config.market,"buy")
                cost += trade_cost
                if not _can_afford(cost, capital):
                    _snap()
                    continue
                pos = "long"
                entry_price = row["Close"]
                entry_atr = row["ATR"]
                stop_loss_price = entry_price - stop_dist
                current_shares = shares
                total_cost = cost
                cumulative_div = 0.0
                entry_date = current_date
                capital -= cost

        else:
            div_income, capital = _collect_dividend(row, (current_shares+current_lots), ps, capital)
            cumulative_div += div_income

            exit_price, exit_reason = _check_long_exit(row, stop_loss_price, config)

            # 加仓（仅在未触发出场信号时检查，以当前 ATR 为基准）
            if not exit_reason and not pd.isna(row["ATR"]) and row["ATR"] > 0:
                units_held = current_shares // base_shares + skip_offset
                if row["Close"] > (entry_price + units_held * 0.5 * row["ATR"]):
                    if current_shares < max_shares:
                        add_price = row["Close"]
                        stop_loss_price = add_price - config.n_value * entry_atr
                        add_cost = add_price * base_shares * ps
                        trade_cost = _pm_trade_cost(add_cost, config.market, "buy")
                        add_cost+= trade_cost
                        if not _can_afford(add_cost, capital):
                            _snap()
                            continue
                        current_shares += base_shares
                        total_cost += add_cost
                        capital -= add_cost
                        more_info.append({
                            "More Price": add_price,
                            "More Date": current_date.strftime("%Y%m%d"),
                        })
                        if skip_offset and len(more_info) == 1:
                            print(f"  ⚠️ [{label}] 已跳过首次加仓（entry + 0.5×ATR），"
                                  f"本次为第2次加仓位")

            if exit_reason:
                sell_proceeds = exit_price * current_shares * ps
                trade_cost = _pm_trade_cost(sell_proceeds, config.market, "sell")
                sell_proceeds-=trade_cost
                pnl = sell_proceeds - total_cost + cumulative_div
                capital += sell_proceeds

                trades.append(_build_trade(
                    entry_date, current_date, "Long",
                    entry_price, exit_price, current_shares,
                    total_cost, pnl, capital, exit_reason,
                    stop_dist=stop_dist, more_info=more_info,
                ))
                pos = None
                current_shares = 0
                base_shares = 0
                max_shares = 0
                cumulative_div = 0.0
                entry_price = 0.0
                entry_atr = 0.0
                stop_loss_price = 0.0
                total_cost = 0.0
                entry_date = None
                more_info = []
                if only_one:
                    _snap()
                    return trades, equity_curve, equity_dates

        _snap()

    # 强制平仓
    if pos == "long":
        last_price = df.iloc[-1]["Close"]
        last_date = df.iloc[-1]["Date"]
        sell_proceeds = last_price * current_shares * ps - _pm_trade_cost(last_price * current_shares * ps, config.market, "sell")
        pnl = sell_proceeds - total_cost + cumulative_div
        capital += sell_proceeds
        last_trade_pnl = pnl  # System 1 输家突破过滤：记录本笔盈亏

        trades.append(_build_trade(
            entry_date, last_date, "Long",
            entry_price, last_price, current_shares,
            total_cost, pnl, capital, "强制平仓",
            stop_dist=stop_dist, more_info=more_info,
        ))
    return trades, equity_curve, equity_dates

# ==================== 加仓做空（金字塔） ====================

def _run_short_with_pyramid(df: pd.DataFrame, config) -> tuple[list[dict], list[float], list]:
    label = config.channel_type
    ps = config.per_shares

    trades = []
    capital = config.initial_capital

    start_index = config.entry_period + 1
    start_date = df.iloc[start_index]["Date"]
    equity_curve = [config.initial_capital]
    equity_dates = [start_date]

    pos = None
    entry_price = 0.0
    entry_atr = 0.0
    stop_loss_price = 0.0
    stop_dist = 0.0
    current_shares = 0
    base_shares = 0
    max_shares = 0
    total_income = 0.0
    entry_date = None
    more_info = []

    for i in range(start_index, len(df)):
        row = df.iloc[i]
        current_date = row["Date"]
        close = float(row["Close"])

        def _snap():
            equity_curve.append(capital - current_shares * close * ps)
            equity_dates.append(current_date)

        if pos is None:
            if pd.isna(row["ATR"]) or row["ATR"] == 0:
                _snap()
                continue
            atr = row["ATR"]
            stop_dist = config.n_value * atr
            # Turtle 1 Unit = 1% Account ÷ (N × Dollars per Point)
            base_shares = int(0.01 * capital / (atr * ps))
            if base_shares <= 0:
                _snap()
                continue
            shares = base_shares
            skip_offset = 1 if config.skip_first_add else 0
            max_shares = shares * (config.max_pyramid - skip_offset)

            if _short_entry_triggered(row, config) and _ema_trend_ok(row, config,"short"):
                entry_price = row["Close"]
                entry_atr = row["ATR"]
                income = entry_price * shares * ps
                pos = "short"
                stop_loss_price = entry_price + stop_dist
                current_shares = shares
                entry_date = current_date
                capital += income
                total_income += income
        else:
            exit_price, exit_reason = _check_short_exit(row, stop_loss_price, config)

            # 加仓（仅在未触发出场信号时检查，以当前 ATR 为基准）
            if not exit_reason and not pd.isna(row["ATR"]) and row["ATR"] > 0:
                units_held = current_shares // base_shares + skip_offset
                if row["Close"] < (entry_price - units_held * 0.5 * row["ATR"]) and _ema_trend_ok(row, config,"short"):
                    if current_shares < max_shares:
                        add_price = row["Close"]
                        stop_loss_price = add_price + config.n_value * entry_atr
                        add_income = add_price * base_shares * ps
                        current_shares += base_shares
                        capital += add_income
                        total_income += add_income
                        more_info.append({
                            "More Price": add_price,
                            "More Date": current_date.strftime("%Y%m%d"),
                        })
                        if skip_offset and len(more_info) == 1:
                            print(f"  ⚠️ [{label}] 已跳过首次加仓（entry + 0.5×ATR），"
                                  f"本次为第2次加仓位")

            if exit_reason:
                sale_proceeds = exit_price * current_shares * ps
                pnl = total_income - sale_proceeds
                capital -= sale_proceeds

                trades.append(_build_trade(
                    entry_date, current_date, "short",
                    entry_price, exit_price, current_shares,
                    total_income, pnl, capital, exit_reason,
                    stop_dist=stop_dist, more_info=more_info,
                ))
                pos = None
                current_shares = 0
                base_shares = 0
                max_shares = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_loss_price = 0.0
                total_income = 0.0
                entry_date = None
                more_info = []

        _snap()

    # 强制平仓
    if pos == "short":
        last_price = df.iloc[-1]["Close"]
        last_date = df.iloc[-1]["Date"]
        sale_proceeds = last_price * current_shares * ps
        pnl = total_income - sale_proceeds
        capital -= sale_proceeds
        last_trade_pnl = pnl  # System 1 输家突破过滤：记录本笔盈亏

        trades.append(_build_trade(
            entry_date, last_date, "short",
            entry_price, last_price, current_shares,
            total_income, pnl, capital, "强制平仓",
            stop_dist=stop_dist, more_info=more_info,
        ))
        equity_curve.append(capital - current_shares * last_price * ps)
        equity_dates.append(last_date)

    return trades, equity_curve, equity_dates


# ==================== 多空双向 ====================

def _run_long_short(df: pd.DataFrame, config) -> tuple[list[dict], list[float], list]:
    label = "System 1"
    PS = 100  # 多空脚本硬编码，不使用 config.per_shares

    trades = []
    capital = config.initial_capital

    start_index = config.entry_period + 1
    start_date = df.iloc[start_index]["Date"]
    equity_curve = [config.initial_capital]
    equity_dates = [start_date]

    pos = None          # None | 'long' | 'short'
    entry_price = 0.0
    stop_loss_price = 0.0
    current_shares = 0  # long 为正, short 为负
    cumulative_div = 0.0
    entry_date = None

    for i in range(start_index, len(df)):
        row = df.iloc[i]
        current_date = row["Date"]
        close = float(row["Close"])

        def _snap():
            # current_shares: +long, -short; 权益 = 现金 + 持仓市值
            equity_curve.append(capital + current_shares * close * PS)
            equity_dates.append(current_date)

        if pos is None:
            stop_dist, shares = _prep_entry(row, capital, config, PS)
            if shares <= 0:
                _snap()
                continue

            entry_price = row["Close"]

            if _long_entry_triggered(row, config) and _ema_trend_ok(row, config):
                cost = entry_price * shares * PS
                if cost > capital * 0.95:
                    _snap()
                    continue
                pos = "long"
                stop_loss_price = entry_price - stop_dist
                current_shares = shares
                cumulative_div = 0.0
                entry_date = current_date
                capital -= cost

            elif _short_entry_triggered(row, config) and _ema_trend_ok(row, config, "short"):
                income = entry_price * shares * PS
                pos = "short"
                stop_loss_price = entry_price + stop_dist
                current_shares = -shares
                cumulative_div = 0.0
                entry_date = current_date
                capital += income

        else:
            if pos == "long":
                div_income, capital = _collect_dividend(row, current_shares, PS, capital)
                cumulative_div += div_income

            exit_price, exit_reason = _check_both_exit(row, pos, stop_loss_price, config)
            if exit_reason:
                pnl = (exit_price - entry_price) * current_shares * PS + cumulative_div
                capital += exit_price * current_shares * PS

                if pos == "long":
                    cost_basis = entry_price * current_shares * PS
                else:
                    cost_basis = abs(exit_price * current_shares * PS)

                trades.append(_build_trade(
                    entry_date, current_date, pos.capitalize(),
                    entry_price, exit_price, current_shares,
                    cost_basis, pnl, capital, exit_reason,
                ))
                pos = None
                current_shares = 0
                cumulative_div = 0.0
                entry_price = 0.0
                stop_loss_price = 0.0
                entry_date = None

        _snap()

    # 强制平仓
    if pos is not None:
        last_price = df.iloc[-1]["Close"]
        last_date = df.iloc[-1]["Date"]
        pnl = (last_price - entry_price) * current_shares * PS + cumulative_div
        sell_proceeds = last_price * current_shares * PS

        if pos == "long":
            cost_basis = entry_price * current_shares * PS
        else:
            cost_basis = abs(last_price * current_shares * PS)
        capital += sell_proceeds

        trades.append(_build_trade(
            entry_date, last_date, pos.capitalize(),
            entry_price, last_price, current_shares,
            cost_basis, pnl, capital, "强制平仓",
        ))
        equity_curve.append(capital + current_shares * last_price * PS)
        equity_dates.append(last_date)

    return trades, equity_curve, equity_dates


# ==================== 反马丁格尔策略 (Anti-Martingale) ====================

def _run_anti_martingale(df: pd.DataFrame, config) -> tuple[list[dict], list[float], list]:
    """反马丁格尔策略：
    - 入场：Close > MA(am_ma_period) 且 EMA 趋势过滤器允许
    - 出场：Close < MA(am_ma_period) 或 硬止损 (N×ATR)
    - 仓位：盈利后下笔风险乘以 am_risk_multiplier，亏损后重置为 1.0
    """
    label = "Anti-Martingale"
    ps = config.per_shares

    trades = []
    capital = config.initial_capital

    start_index = max(config.am_ma_period, config.atr_period) + 1
    start_date = df.iloc[start_index]["Date"]
    equity_curve = [config.initial_capital]
    equity_dates = [start_date]

    risk_multiplier = 1.0

    pos = None
    entry_price = 0.0
    stop_loss_price = 0.0
    stop_dist = 0.0
    current_shares = 0
    cumulative_div = 0.0
    entry_date = None

    for i in range(start_index, len(df)):
        row = df.iloc[i]
        current_date = row["Date"]
        close = float(row["Close"])

        def _snap():
            equity_curve.append(capital + current_shares * close * ps)
            equity_dates.append(current_date)

        if pos is None:
            if pd.isna(row["ATR"]) or row["ATR"] == 0:
                _snap()
                continue
            atr = row["ATR"]
            stop_dist = config.n_value * atr

            effective_risk = config.risk_per_trade * risk_multiplier
            shares = _calc_shares(capital, effective_risk, stop_dist, ps)
            if shares <= 0:
                _snap()
                continue

            if _am_entry_triggered(row, config) and _ema_trend_ok(row, config):
                cost = close * shares * ps
                if not _can_afford(cost, capital):
                    _snap()
                    continue
                pos = "long"
                entry_price = close
                stop_loss_price = entry_price - stop_dist
                current_shares = shares
                cumulative_div = 0.0
                entry_date = current_date
                capital -= cost

        else:
            div_income, capital = _collect_dividend(row, current_shares, ps, capital)
            cumulative_div += div_income

            exit_price = None
            exit_reason = ""

            if close < stop_loss_price:
                exit_price = close
                exit_reason = "Hard Stop Loss (N×ATR)"

            if not exit_reason and _am_exit_triggered(row, config):
                exit_price = close
                exit_reason = "Trend Reversal (MA)"

            if exit_reason:
                cost_basis = entry_price * current_shares * ps
                pnl = (exit_price - entry_price) * current_shares * ps + cumulative_div
                capital += exit_price * current_shares * ps

                trades.append(_build_trade(
                    entry_date, current_date, "Long",
                    entry_price, exit_price, current_shares,
                    cost_basis, pnl, capital, exit_reason,
                    stop_dist=stop_dist,
                ))

                if pnl > 0:
                    risk_multiplier = min(
                        risk_multiplier * config.am_risk_multiplier,
                        config.am_max_multiplier,
                    )
                else:
                    risk_multiplier = 1.0

                pos = None
                current_shares = 0
                cumulative_div = 0.0
                entry_price = 0.0
                stop_loss_price = 0.0
                entry_date = None

        _snap()

    # 强制平仓
    if pos == "long":
        last_price = df.iloc[-1]["Close"]
        last_date = df.iloc[-1]["Date"]
        cost_basis = entry_price * current_shares * ps
        pnl = (last_price - entry_price) * current_shares * ps + cumulative_div
        capital += last_price * current_shares * ps

        trades.append(_build_trade(
            entry_date, last_date, "Long",
            entry_price, last_price, current_shares,
            cost_basis, pnl, capital, "强制平仓",
            stop_dist=stop_dist,
        ))
        equity_curve.append(capital + current_shares * last_price * ps)
        equity_dates.append(last_date)

    return trades, equity_curve, equity_dates


# ==================== 仓位管理策略1 (Position Management 1) ====================

def _pm_trade_cost(amount: float, market: str, trade_type: str) -> float:
    """计算仓位管理交易成本。
    amount: 成交金额（价格 × 手数 × 每手股数）
    market: "A" | "HK"
    trade_type: "buy" | "sell"
    """
    if market == "HK":
        # 港股：佣金0.01% + 印花税0.1% + 其他0.01% = 0.12%
        return amount * 0.0012
    else:  # A股
        if trade_type == "buy":
            # A股买入：佣金0.01% + 印花税0.05% = 0.06%
            return amount * 0.0006
        else:
            # A股卖出：佣金0.01%
            return amount * 0.0001


def _run_position_management_1(df: pd.DataFrame, config) -> tuple[list[dict], list[float], list]:
    """仓位管理策略1 - 网格仓位管理：
    - 从 pm_start_date 开始，持有 pm_initial_shares 手初始仓位
    - 以起始日收盘价为参考价
    - 相对参考价上涨 pm_grid_up → 减仓 pm_reduce_pct（权益%）
    - 相对参考价下跌 pm_grid_down → 加仓 pm_add_pct（权益%）
    - 每次交易后参考价重置为成交价，形成动态网格
    - 仓位上限 pm_max_position_pct，现金下限 pm_min_cash_pct 防止过度加仓
    """
    ps = config.per_shares
    initial_lots = config.pm_initial_shares
    reduce_pct = config.pm_reduce_pct
    add_pct = config.pm_add_pct
    grid_up = config.pm_grid_up
    grid_down = config.pm_grid_down
    max_pos_pct = config.pm_max_position_pct
    min_cash_pct = config.pm_min_cash_pct
    dynamic_pos = config.pm_dynamic_position  # 动态仓位上限开关
    dyn_v1 = config.pm_dyn_vol_1              # 波动率分档阈值
    dyn_v2 = config.pm_dyn_vol_2
    dyn_v3 = config.pm_dyn_vol_3
    dyn_p1 = config.pm_dyn_pos_1              # 各档仓位上限
    dyn_p2 = config.pm_dyn_pos_2
    dyn_p3 = config.pm_dyn_pos_3
    dyn_p4 = config.pm_dyn_pos_4
    take_profit = config.pm_take_profit_ratio # 止盈回本倍数
    dmi_bl = df.attrs.get("dmi_baseline")      # 个股 DMI 基线（自动适配，无需配置）
    best_zone = _get_best_zone(config.ts_code)  # 个股最优 DMI 区域（zone_score 权重最高者）

    pm_trades = []
    capital = config.initial_capital
    current_lots = initial_lots

    # 定位起始日期
    start_date_str = config.pm_start_date
    if not start_date_str:
        print("⚠️ [仓位管理] 未设置 pm_start_date，跳过")
        return pm_trades, [config.initial_capital]

    start_date = pd.to_datetime(start_date_str)
    df_dates = pd.to_datetime(df["Date"])
    mask = df_dates >= start_date
    if not mask.any():
        print(f"⚠️ [仓位管理] 数据中没有 >= {start_date_str} 的日期，跳过")
        return pm_trades, [config.initial_capital]
    start_index = int(np.argmax(mask.values))

    # 初始化参考价为起始日收盘价
    start_row = df.iloc[start_index]
    reference_price = float(start_row["Close"])

    # 初始权益
    initial_equity = capital + current_lots * reference_price * ps
    equity_curve = [initial_equity]
    equity_dates = [start_date]          # 与 equity_curve 一一对应的日期序列
    donchian_exit_date = None
    for i in range(start_index + 1, len(df)):
        row = df.iloc[i]
        close = float(row["Close"])
        current_date = row["Date"]

        def _snap():
            """记录当日收盘权益快照"""
            equity_curve.append(capital + current_lots * close * ps)
            equity_dates.append(current_date)

        if donchian_exit_date:
            if donchian_exit_date >= current_date:
                continue
        equity_before = capital + current_lots * close * ps

        eff_grid_down, eff_grid_up, eff_reduce = _calc_effective_grid(row, config, dmi_bl)

        hh_entry = row.get("HH_Entry", np.nan)
        # DMI + 趋势自适应（数据实证优化：A区加速+B区中继都可追涨，B区20日+6.98%优于A区+4.58%）
        donchian_done = False
        if config.pm_adaptive and close > hh_entry and _dmi_trend_area(row, config, "long", dmi_bl) == "A":
            # ---- 单边上涨趋势 → Donchian 通道：趋势跟踪，不轻易卖 ----
            # Donchian 买入：突破20日高点 + 趋势加速/中继确认 → 加仓
            df_donchian = df.iloc[i:]
            tmp_config = copy.deepcopy(config)
            tmp_config.initial_capital = capital
            tmp_config.don_start_date = current_date
            trades,eq,eq_dates =_run_long_with_pyramid(df_donchian, tmp_config, True,current_lots)
            if trades:
                donchian_done = True
                add_lots = trades[0].get("Shares",0)
                total_cost = trades[0].get("Cost_¥",0)
                more_info = trades[0].get("More Info",[])
                # capital -= total_cost
                capital = trades[0].get("Capital_After")
                exit_price = trades[0].get("Exit Price")
                # 平仓交易金额
                trade_amount = exit_price * add_lots * ps - _pm_trade_cost(exit_price * add_lots * ps,config.market,"sell")
                # donchian加仓后的现金=donchian平仓后的现金-平仓交易金额
                capital -= trade_amount
                current_lots += add_lots
                equity = capital + current_lots * close * ps
                reference_price = close
                pm_trades.append({
                    "Date": current_date,
                    "Entry Date": current_date,
                    "Exit Date": current_date,
                    "Type": "加仓",
                    "Price": close,
                    "Shares": add_lots,
                    "Entry Price": close,
                    "Exit Price": close,
                    "Cost_¥": 0,
                    "PnL_¥": 0,
                    "PnL_%": 0.0,
                    "Capital_After": capital,
                    "Reason": f"Donchian买入(突破{int(hh_entry)}，趋势延续)",
                    "Amount": -total_cost,
                    "Position_After": current_lots,
                    "Equity_After": equity,
                    "Reference_Price": reference_price,
                    "more_info": more_info,
                    "+DI":row["+DI"],
                    "-DI":row["-DI"],
                    "ADX14":row["ADX14"],
                    "ADXR":row["ADXR"],
                    "ADX_UP": row["ADX_UP"],
                    "ADX_CROSS_UP_ADXR": row["ADX_CROSS_UP_ADXR"],
                    "ADX_DOWN": row["ADX_DOWN"],
                    "ADX_CROSS_DOWN_ADXR": row["ADX_CROSS_DOWN_ADXR"],
                })
                donchian_exit_date = trades[0].get("Exit Date")
                capital = trades[0].get("Capital_After")
                reason = trades[0].get("Reason")
                if reason !="强制平仓": #强制平仓不记录减仓
                    current_lots -= add_lots
                    equity = capital + current_lots * exit_price * ps
                    reference_price = exit_price
                    pm_trades.append({
                        "Date": donchian_exit_date,
                        "Entry Date": donchian_exit_date,
                        "Exit Date": donchian_exit_date,
                        "Type": "减仓",
                        "Price": exit_price,
                        "Shares": add_lots,
                        "Entry Price": exit_price,
                        "Exit Price": exit_price,
                        "Cost_¥": 0,
                        "PnL_¥": 0,
                        "PnL_%": 0.0,
                        "Capital_After": capital,
                        "Reason": reason,
                        "Amount": trade_amount,
                        "Position_After": current_lots,
                        "Equity_After": equity,
                        "Reference_Price": reference_price,
                        "+DI": row["+DI"],
                        "-DI": row["-DI"],
                        "ADX14": row["ADX14"],
                        "ADXR": row["ADXR"],
                        "ADX_UP": row["ADX_UP"],
                        "ADX_CROSS_UP_ADXR": row["ADX_CROSS_UP_ADXR"],
                        "ADX_DOWN": row["ADX_DOWN"],
                        "ADX_CROSS_DOWN_ADXR": row["ADX_CROSS_DOWN_ADXR"],
                    })
                # 趋势反转出场 → 趋势已破，清空剩余网格仓位
                # 止损出场 → 可能是假突破，保留网格仓位继续运作
                is_trend_reversal = "Trend Reversal" in reason
                if current_lots > 0 and is_trend_reversal:
                    #同时清仓剩余仓位
                    reduce_lots = current_lots
                    gross_proceeds = exit_price * reduce_lots * ps
                    trade_cost = _pm_trade_cost(gross_proceeds, config.market, "sell")
                    net_proceeds = gross_proceeds - trade_cost
                    capital += net_proceeds
                    current_lots=0
                    reference_price = exit_price
                    equity = capital
                    pm_trades.append({
                        "Date": donchian_exit_date,
                        "Entry Date": donchian_exit_date,
                        "Exit Date": donchian_exit_date,
                        "Type": "减仓",
                        "Price": exit_price,
                        "Shares": reduce_lots,
                        "Entry Price": exit_price,
                        "Exit Price": exit_price,
                        "Cost_¥": 0,
                        "PnL_¥": 0,
                        "PnL_%": 0.0,
                        "Capital_After": capital,
                        "Reason": reason,
                        "Amount": net_proceeds,
                        "Position_After": current_lots,
                        "Equity_After": equity,
                        "Reference_Price": reference_price,
                        "+DI": row["+DI"],
                        "-DI": row["-DI"],
                        "ADX14": row["ADX14"],
                        "ADXR": row["ADXR"],
                        "ADX_UP": row["ADX_UP"],
                        "ADX_CROSS_UP_ADXR": row["ADX_CROSS_UP_ADXR"],
                        "ADX_DOWN": row["ADX_DOWN"],
                        "ADX_CROSS_DOWN_ADXR": row["ADX_CROSS_DOWN_ADXR"],
                    })
                    eq[-1] = equity
                equity_curve.extend(eq)
                equity_dates.extend(eq_dates)
        # ---- Donchian 未触发或失败 → 尝试网格交易 ----
        if not donchian_done:
            if current_lots > 0:
                div_income, capital = _collect_dividend(row, current_lots, ps, capital)
            # 空仓时市价已涨超一个卖出网格→追认当前价格，避免牛市踏空，
            # config.pm_adaptive为false时不重置参考价，减少交易次数，且没有存在价格回不到之前的网格的情况
            if config.pm_adaptive and current_lots == 0 and close > reference_price * (1 + eff_grid_up):
                reference_price = close
            # 非单边上涨/下跌时网格交易买入
            # 失效模式④修复：short C区(下跌酝酿)最危险(20日+0.52%)，不应允许买入
            # 实证：short A/B(下跌确认,20d+0.32%/-0.23%)拦截买入；
            # long C(趋势酝酿,20d仅+1.09%/胜46.6%)也拦截，等趋势确认后再入场
            if (close <= reference_price * (1 - eff_grid_down)
              and _dmi_trend_area(row, config, "short", dmi_bl) != best_zone
              and _dmi_trend_area(row, config, "long", dmi_bl) != "C"):
                add_value = equity_before * add_pct
                add_lots = math.floor(add_value / (close * ps))
                if add_lots <= 0:
                    _snap()
                    continue
                if current_lots == 0 and equity_before > initial_equity * take_profit:
                    # 当前已回本停止交易
                    _snap()
                    continue
                # 仓位上限（动态：根据波动率 ATR/价格 自动调整）
                if dynamic_pos and not pd.isna(row["ATR"]) and row["ATR"] > 0:
                    vol = row["ATR"] / close
                    if vol < dyn_v1:
                        effective_max_pos = dyn_p1    # 低波动
                    elif vol < dyn_v2:
                        effective_max_pos = dyn_p2    # 中波动
                    elif vol < dyn_v3:
                        effective_max_pos = dyn_p3    # 高波动
                    else:
                        effective_max_pos = dyn_p4    # 极端波动
                else:
                    effective_max_pos = max_pos_pct
                position_value_after = (current_lots + add_lots) * close * ps
                if position_value_after > equity_before * effective_max_pos:
                    add_lots = int((equity_before * effective_max_pos) / (close * ps)) - current_lots
                    if add_lots <= 0:
                        _snap()
                        continue

                # 现金下限
                gross_cost = close * add_lots * ps
                trade_cost = _pm_trade_cost(gross_cost, config.market, "buy")
                total_cost = gross_cost + trade_cost
                cash_after = capital - total_cost
                if cash_after < equity_before * min_cash_pct:
                    max_cost = capital - equity_before * min_cash_pct
                    afford_per_lot = close * ps * (1 + (0.0012 if config.market == "HK" else 0.0006))
                    add_lots = int(max_cost / afford_per_lot) if afford_per_lot > 0 else 0
                if add_lots <= 0:
                    _snap()
                    continue
                gross_cost = close * add_lots * ps
                trade_cost = _pm_trade_cost(gross_cost, config.market, "buy")
                total_cost = gross_cost + trade_cost

                capital -= total_cost
                current_lots += add_lots
                reference_price = close

                equity = capital + current_lots * close * ps
                pm_trades.append({
                    "Date": current_date,
                    "Entry Date": current_date,
                    "Exit Date": current_date,
                    "Type": "加仓",
                    "Price": close,
                    "Entry Price": close,
                    "Exit Price": close,
                    "Shares": add_lots,
                    "Cost_¥": trade_cost,
                    "PnL_¥": -total_cost,
                    "PnL_%": 0.0,
                    "Capital_After": capital,
                    "Reason": f"网格买入 -{round(eff_grid_down * 100,2)}%（权益{round(add_pct * 100,2)}%）",
                    "Amount": -total_cost,
                    "Position_After": current_lots,
                    "Equity_After": equity,
                    "Reference_Price": reference_price,
                    "more_info":[],
                    "+DI": row["+DI"],
                    "-DI": row["-DI"],
                    "ADX14": row["ADX14"],
                    "ADXR": row["ADXR"],
                    "ADX_UP": row["ADX_UP"],
                    "ADX_CROSS_UP_ADXR": row["ADX_CROSS_UP_ADXR"],
                    "ADX_DOWN": row["ADX_DOWN"],
                    "ADX_CROSS_DOWN_ADXR": row["ADX_CROSS_DOWN_ADXR"],
                })
                # 每日收盘权益快照（不论是否发生交易）
                _snap()
            else:
                # ---- 非单边上涨趋势 → 网格交易 ----
                # 失效模式③修复：B区20日+6.98%优于A区+4.58%，卖出拦截从仅A区扩展到A+B
                if (close >= reference_price * (1 + eff_grid_up)
                        and current_lots > 0
                        and _dmi_trend_area(row, config, "long", dmi_bl) != best_zone):
                    reduce_value = equity_before * eff_reduce
                    reduce_lots = math.ceil(reduce_value / (close * ps))
                    reduce_lots = min(reduce_lots, current_lots)
                    if  equity_before > initial_equity * take_profit:
                        # 当前已回本则全部清仓
                        reduce_lots=current_lots
                    if reduce_lots > 0:
                        gross_proceeds = close * reduce_lots * ps
                        trade_cost = _pm_trade_cost(gross_proceeds, config.market, "sell")
                        net_proceeds = gross_proceeds - trade_cost
                        capital += net_proceeds
                        current_lots -= reduce_lots
                        reference_price = close

                        equity = capital + current_lots * close * ps
                        pm_trades.append({
                            "Date": current_date, "Entry Date": current_date,
                            "Exit Date": current_date, "Type": "减仓",
                            "Price": close, "Entry Price": close, "Exit Price": close,
                            "Shares": reduce_lots, "Cost_¥": trade_cost,
                            "PnL_¥": net_proceeds, "PnL_%": 0.0,
                            "Capital_After": capital,
                            "Reason": f"网格卖出 +{round(eff_grid_up*100,2)}%（权益{round(eff_reduce*100,2)}%）",
                            "Amount": net_proceeds, "Position_After": current_lots,
                            "Equity_After": equity, "Reference_Price": reference_price,
                            "more_info": [],
                            "+DI": row["+DI"],
                            "-DI": row["-DI"],
                            "ADX14": row["ADX14"],
                            "ADXR": row["ADXR"],
                            "ADX_UP": row["ADX_UP"],
                            "ADX_CROSS_UP_ADXR": row["ADX_CROSS_UP_ADXR"],
                            "ADX_DOWN": row["ADX_DOWN"],
                            "ADX_CROSS_DOWN_ADXR": row["ADX_CROSS_DOWN_ADXR"],
                        })

                # 每日收盘权益快照（不论是否发生交易）
                _snap()

    # 追加最终权益点（与最后一天一致，_snap 已记录）
    # 注意：循环中 _snap() 已经覆盖了最后一天，此处无需重复
    # 但如果 start_index+1 > len(df)-1（无交易日），则需保留初始点

    return pm_trades, equity_curve, equity_dates


# ==================== 仓位管理策略2 (Position Management 2) ====================

def _run_position_management_2(df: pd.DataFrame, config) -> tuple[list[dict], list[float], list]:
    """仓位管理策略2 - Swing 恢复仓位管理：
    - 从 pm_start_date 开始，持有 pm_initial_shares 手初始仓位
    - 状态机：holding ↔ waiting_buyback
    - holding 状态：跟踪近期低点，相对低点反弹 pm2_bounce_up → 卖出 pm2_trade_lots 手
    - waiting_buyback 状态：跟踪卖出价，相对卖出价回落 pm2_buyback_down → 买回 pm2_trade_lots 手
    - 卖出和买回的振幅不对等（bounce_up > buyback_down 才有差价利润）
    - 每次固定手数交易，靠反复 swing 赚取差价摊薄持仓成本
    - 仓位上限 pm_max_position_pct，现金下限 pm_min_cash_pct
    """
    ps = config.per_shares
    initial_lots = config.pm_initial_shares
    bounce_up = config.pm2_bounce_up
    buyback_down = config.pm2_buyback_down
    trade_lots = config.pm2_trade_lots
    max_pos_pct = config.pm_max_position_pct
    min_cash_pct = config.pm_min_cash_pct

    pm_trades = []
    capital = config.initial_capital
    current_lots = initial_lots

    # 定位起始日期
    start_date_str = config.pm_start_date
    if not start_date_str:
        print("⚠️ [仓位管理2] 未设置 pm_start_date，跳过")
        return pm_trades, [config.initial_capital]

    start_date = pd.to_datetime(start_date_str)
    df_dates = pd.to_datetime(df["Date"])
    mask = df_dates >= start_date
    if not mask.any():
        print(f"⚠️ [仓位管理2] 数据中没有 >= {start_date_str} 的日期，跳过")
        return pm_trades, [config.initial_capital]
    start_index = int(np.argmax(mask.values))

    # 初始化：近期低点 = 起始日收盘价，状态 = holding
    start_row = df.iloc[start_index]
    recent_low = float(start_row["Close"])
    last_sell_price = 0.0
    state = "holding"

    # 初始权益
    initial_equity = capital + current_lots * recent_low * ps
    equity_curve = [initial_equity]
    equity_dates = [start_date]

    for i in range(start_index + 1, len(df)):
        row = df.iloc[i]
        close = float(row["Close"])
        current_date = row["Date"]

        def _snap():
            """记录当日收盘权益快照"""
            equity_curve.append(capital + current_lots * close * ps)
            equity_dates.append(current_date)

        equity_before = capital + current_lots * close * ps

        # 无论什么状态，都更新近期低点（用于后续卖出参考）
        if close < recent_low:
            recent_low = close

        if state == "holding":
            # ---- 反弹触发卖出 ----
            bounce_trigger = recent_low * (1 + bounce_up)
            if close >= bounce_trigger and current_lots >= trade_lots:
                sell_lots = min(trade_lots, current_lots)
                if sell_lots <= 0:
                    _snap()
                    continue

                gross = close * sell_lots * ps
                trade_cost = _pm_trade_cost(gross, config.market, "sell")
                net = gross - trade_cost
                capital += net
                current_lots -= sell_lots

                equity = capital + current_lots * close * ps
                pm_trades.append({
                    "Date": current_date, "Entry Date": current_date,
                    "Exit Date": current_date, "Type": "减仓",
                    "Price": close, "Entry Price": close, "Exit Price": close,
                    "Shares": sell_lots, "Cost_¥": trade_cost,
                    "PnL_¥": net, "PnL_%": 0.0,
                    "Capital_After": capital,
                    "Reason": f"Swing卖出(反弹+{int(bounce_up*100)}%)",
                    "Amount": net, "Position_After": current_lots,
                    "Equity_After": equity, "Reference_Price": close,
                })

                last_sell_price = close
                recent_low = close      # 重置低点为卖出价
                state = "waiting_buyback"

        elif state == "waiting_buyback":
            # ---- 回落触发买回 ----
            buyback_trigger = last_sell_price * (1 - buyback_down)
            if close <= buyback_trigger:
                buy_lots = trade_lots

                # 仓位上限检查
                position_value_after = (current_lots + buy_lots) * close * ps
                if position_value_after > equity_before * max_pos_pct:
                    buy_lots = int((equity_before * max_pos_pct) / (close * ps)) - current_lots
                    if buy_lots <= 0:
                        # 仓位已满，放弃买回，回到 holding
                        state = "holding"
                        _snap()
                        continue

                # 现金下限检查
                gross_cost = close * buy_lots * ps
                trade_cost = _pm_trade_cost(gross_cost, config.market, "buy")
                total_cost = gross_cost + trade_cost
                cash_after = capital - total_cost
                if cash_after < equity_before * min_cash_pct:
                    max_cost = capital - equity_before * min_cash_pct
                    fee_rate = 0.0012 if config.market == "HK" else 0.0006
                    afford_per_lot = close * ps * (1 + fee_rate)
                    buy_lots = int(max_cost / afford_per_lot) if afford_per_lot > 0 else 0
                if buy_lots <= 0:
                    # 现金不足，放弃买回，回到 holding
                    state = "holding"
                    _snap()
                    continue

                gross_cost = close * buy_lots * ps
                trade_cost = _pm_trade_cost(gross_cost, config.market, "buy")
                total_cost = gross_cost + trade_cost

                capital -= total_cost
                current_lots += buy_lots

                equity = capital + current_lots * close * ps
                pm_trades.append({
                    "Date": current_date, "Entry Date": current_date,
                    "Exit Date": current_date, "Type": "加仓",
                    "Price": close, "Entry Price": close, "Exit Price": close,
                    "Shares": buy_lots, "Cost_¥": trade_cost,
                    "PnL_¥": -total_cost, "PnL_%": 0.0,
                    "Capital_After": capital,
                    "Reason": f"Swing买回(回落{int(buyback_down*100)}%)",
                    "Amount": -total_cost, "Position_After": current_lots,
                    "Equity_After": equity, "Reference_Price": close,
                })

                recent_low = close
                state = "holding"

        # 每日收盘权益快照（不论是否发生交易）
        _snap()

    return pm_trades, equity_curve, equity_dates
