#!/usr/bin/env python3
"""模拟未来252天（约1年）多种K线形态 + 仓位管理回测

核心改进：
  - 根据个股历史对应时间段（同日历月份±1）特征来生成模拟K线
  - 支持多股票批量生成，分开保存
  - 每只股票使用自身历史数据匹配场景，捕捉季节性波动特征

生成 8 类模拟数据：
  1. 单边上涨     — 趋势持续向上，低波动
  2. 单边下跌     — 趋势持续向下，低波动
  3. 震荡向上     — 高波动 + 正弦摆动，重心上移
  4. 震荡向下     — 高波动 + 大幅正弦摆动，重心缓慢下移
  5. 横盘震荡     — 价格在箱体内反复，重心几乎不变（网格策略最适场景）
  6. V形反转      — 先急跌再急涨，回到原点附近
  7. 倒V反转      — 先急涨再急跌，回到原点附近
  8. 缓慢阴跌     — 极缓慢下跌，夹杂小反弹（温水煮青蛙）

用法:
  python simulate_market_scenarios.py --all          # 全部5只股票，分开保存
  python simulate_market_scenarios.py                # 单股票模式（向后兼容）
  python simulate_market_scenarios.py --ts-code 02015.HK  # 指定股票
"""

import sys
import os
from datetime import timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # project root

from core.config import BacktestConfig
from core.indicators import compute_indicators
from core.engine import run_backtest
from core.statistics import calc_stats
import os



# ═══════════════════════════════════════════════════════════════
# 配置 — key 为 ts_code，通过 init_config(ts_code) 激活
# ═══════════════════════════════════════════════════════════════
TRADING_DAYS = 252  # 约1年交易日
stock_dir = None
_ALL_BASE_KWARGS = {
    "06181.HK": dict(  # 老铺
        ts_code="06181.HK", stock_name="老铺", market="HK",
        mode="position_management_1", pm_start_date="20260629",
        ema_slow=0, ema_fast=0, pm_initial_shares=3,
        initial_capital=40000, user_dmi=True,
        pm_dynamic_position=False, pm_adaptive_grid=False,
        per_shares=100, data_source="akshare",
    ),
    "02015.HK": dict(  # 理想
        ts_code="02015.HK", stock_name="理想", market="HK",
        mode="position_management_1", pm_start_date="20260629",
        ema_slow=0, ema_fast=0, pm_initial_shares=25,
        initial_capital=40000, user_dmi=True,
        pm_dynamic_position=False, pm_adaptive_grid=False,
        per_shares=100, data_source="akshare",
    ),
    "09992.HK": dict(  # 泡泡玛特
        ts_code="09992.HK", stock_name="泡泡玛特", market="HK",
        mode="position_management_1", pm_start_date="20260629",
        ema_slow=0, ema_fast=0, pm_initial_shares=2,
        initial_capital=40000, user_dmi=True,
        pm_dynamic_position=False, pm_adaptive_grid=True,
        per_shares=200, data_source="akshare",
    ),
    "03690.HK": dict(  # 美团
        ts_code="03690.HK", stock_name="美团", market="HK",
        mode="position_management_1", pm_start_date="20260629",
        ema_slow=0, ema_fast=0, pm_initial_shares=2,
        initial_capital=40000, user_dmi=True,
        pm_dynamic_position=False, pm_adaptive_grid=False,
        per_shares=100, data_source="akshare",
    ),
    "01810.HK": dict(  # 小米
        ts_code="01810.HK", stock_name="小米", market="HK",
        mode="position_management_1", pm_start_date="20260629",
        ema_slow=0, ema_fast=0, pm_initial_shares=8,
        initial_capital=40000, user_dmi=True,
        pm_dynamic_position=False, pm_adaptive_grid=False,
        per_shares=100, data_source="akshare",
    ),

}

_ALL_BEST_PARAMS = {
    "06181.HK": dict(  # 老铺
        pm_grid_down=0.075, pm_grid_up=0.07, pm_reduce_pct=0.38,
        pm_add_pct=0.25, pm_adx_threshold=32, pm_agg_k=1,
    ),
    "02015.HK": dict(  # 理想
        pm_grid_down=0.095, pm_grid_up=0.07, pm_reduce_pct=0.35,
        pm_add_pct=0.30, pm_adx_threshold=25, pm_agg_k=1,
    ),
    "09992.HK": dict(  # 泡泡玛特
        pm_grid_down=0.085, pm_grid_up=0.075, pm_reduce_pct=0.65,
        pm_add_pct=0.52, pm_adx_threshold=21, pm_agg_k=0.66,
    ),
    "03690.HK": dict(  # 美团
        pm_grid_down=0.10, pm_grid_up=0.027, pm_reduce_pct=0.57,
        pm_add_pct=0.52, pm_adx_threshold=35, pm_agg_k=1,
    ),
    "01810.HK": dict(  # 小米
        pm_grid_down=0.07, pm_grid_up=0.05, pm_reduce_pct=0.69,
        pm_add_pct=0.33, pm_adx_threshold=26, pm_agg_k=1,
    ),
}

# 运行时激活的配置（由 init_config 设置）
BASE_KWARGS = None
BEST_PARAMS = None


def init_config(ts_code: str):
    """根据 ts_code 激活对应的 BASE_KWARGS 和 BEST_PARAMS。"""
    global BASE_KWARGS, BEST_PARAMS
    if ts_code not in _ALL_BASE_KWARGS:
        raise KeyError(f"未配置 {ts_code} 的 BASE_KWARGS，请在 _ALL_BASE_KWARGS 中添加")
    if ts_code not in _ALL_BEST_PARAMS:
        raise KeyError(f"未配置 {ts_code} 的 BEST_PARAMS，请在 _ALL_BEST_PARAMS 中添加")
    BASE_KWARGS = _ALL_BASE_KWARGS[ts_code]
    BEST_PARAMS = _ALL_BEST_PARAMS[ts_code]





def get_last_price() -> float:
    """获取理想汽车最近一个交易日的收盘价"""
    from core.cache import load as cache_load
    cached = cache_load(BASE_KWARGS.get("ts_code"), BASE_KWARGS.get("market"))
    if cached is not None and not cached.empty:
        return float(cached["Close"].iloc[-1]),cached["Date"].iloc[-1]
    raise ValueError(f"{BASE_KWARGS.get("ts_code")}未获取到缓存日线行情")


def setup_cjk_font():
    """查找并设置中文字体"""
    try:
        import matplotlib
        import matplotlib.font_manager as fm
        candidates = ["PingFang SC", "Heiti SC", "STHeiti",
                      "PingFang HK", "Arial Unicode MS", "Hiragino Sans GB"]
        for name in candidates:
            for f in fm.fontManager.ttflist:
                if f.name == name:
                    print(f"🔤 使用字体: {f.name}")
                    matplotlib.rcParams["font.family"] = [name, "sans-serif"]
                    matplotlib.rcParams["axes.unicode_minus"] = False
                    return
        print("⚠️ 未找到中文字体，使用英文标签")
    except Exception:
        print("⚠️ 字体设置失败")


# ═══════════════════════════════════════════════════════════════
# 场景目标特征 (总变化%, 日波动%, 涨日比%) — 天数由 generate_ohlcv 的 days 参数决定
# ═══════════════════════════════════════════════════════════════
_SCENARIO_TARGET = {
    "up":              (65.0, 1.40, 58),
    "down":            (-40.0, 1.40, 35),
    "volatile_up":     (21.0, 2.50, 50),
    "volatile_down":   (-12.0, 2.60, 42),
    "sideways":        (0.0, 1.80, 50),
    "v_reversal":      (0.0, 1.80, 48),
    "inv_v_reversal":  (0.0, 1.80, 52),
    "slow_grind_down": (-20.0, 1.50, 40),
}


def _find_best_historical_window(close: np.ndarray, target_chg: float,
                                 target_vol: float, target_up: float,
                                 window: int = 126, step: int = 10,
                                 dates_full: np.ndarray = None,
                                 start_month: int = None) -> tuple:
    """扫描历史，找到与目标特征最相似的 window 天区间。

    如果提供 start_month，仅考虑起始月份在 [start_month-1, start_month+1]
    范围内的历史窗口（含跨年回绕），以捕捉季节性特征。
    """
    best_score, best = -1.0, None
    for s in range(0, len(close) - window, step):
        # 季节性过滤：仅匹配同日历月份(±1)的历史窗口
        if start_month is not None and dates_full is not None:
            win_start_month = pd.Timestamp(dates_full[s]).month
            month_diff = abs(win_start_month - start_month)
            if month_diff > 6:
                month_diff = 12 - month_diff  # 跨年回绕 (12月↔1月)
            if month_diff > 1:
                continue
        e = s + window
        win = close[s:e]
        rets = np.diff(win) / win[:-1]
        chg = (win[-1] / win[0] - 1) * 100
        vol = np.std(rets) * 100
        up = (rets > 0).sum() / len(rets) * 100
        score = (max(0.0, 1.0 - abs(chg - target_chg) / 50) * 0.4
                 + max(0.0, 1.0 - abs(vol - target_vol) / 2.0) * 0.3
                 + max(0.0, 1.0 - abs(up - target_up) / 25) * 0.3)
        if score > best_score:
            best_score = score
            best = (s, e, chg, vol, up)
    if best is None:
        # 季节性过滤太严导致无匹配 → 回退到全量扫描
        for s in range(0, len(close) - window, step):
            e = s + window
            win = close[s:e]
            rets = np.diff(win) / win[:-1]
            chg = (win[-1] / win[0] - 1) * 100
            vol = np.std(rets) * 100
            up = (rets > 0).sum() / len(rets) * 100
            score = (max(0.0, 1.0 - abs(chg - target_chg) / 50) * 0.4
                     + max(0.0, 1.0 - abs(vol - target_vol) / 2.0) * 0.3
                     + max(0.0, 1.0 - abs(up - target_up) / 25) * 0.3)
            if score > best_score:
                best_score = score
                best = (s, e, chg, vol, up)
    return (*best, best_score)


def _bootstrap_returns(close: np.ndarray, s_idx: int, e_idx: int,
                       target_slope: float, n_days: int,
                       rng: np.random.Generator,
                       v_reverse: bool = False) -> np.ndarray:
    """从历史区间提取日收益 → 打乱 → 调整均值 → 生成价格路径（从1.0起）。"""
    hist_rets = np.diff(close[s_idx:e_idx]) / close[s_idx:e_idx - 1]
    shuffled = rng.choice(hist_rets, size=n_days, replace=True)
    if v_reverse:
        # V形/倒V：后半段取反
        half = n_days // 2
        adjusted = np.concatenate([
            shuffled[:half] + target_slope - np.mean(shuffled[:half]),
            -shuffled[half:] + target_slope - np.mean(-shuffled[half:]),
        ])
    else:
        adjusted = shuffled + target_slope - np.mean(shuffled)
    return np.cumprod(1 + adjusted), adjusted


def _wrap_ohlcv(dates: pd.DatetimeIndex, close_path: np.ndarray,
                start_price: float, rng: np.random.Generator,
                ts_code: str = "09992.HK") -> pd.DataFrame:
    """给定 Close 序列，生成完整 OHLCV DataFrame。"""
    n = len(close_path)
    prev = np.concatenate([[start_price], close_path[:-1]])
    daily_ret = close_path / prev - 1
    daily_vol = np.abs(daily_ret) + 0.005
    opens = np.zeros(n)
    highs = np.zeros(n)
    lows = np.zeros(n)
    for i in range(n):
        c = close_path[i]
        gap = rng.normal(0, daily_vol[i] * 0.3)
        o = prev[i] * (1 + gap)
        spread = abs(rng.normal(0, daily_vol[i] * 0.5))
        highs[i] = max(o, c) + spread
        lows[i] = min(o, c) - spread
        if lows[i] <= 0:
            lows[i] = min(o, c) * 0.98
        opens[i] = o
    avg_vol = 50000
    volumes = rng.lognormal(np.log(avg_vol), 0.4, n).astype(int)
    amounts = (volumes * close_path * 100 / 1000).round(2)
    return pd.DataFrame({
        "ts_code": ts_code,
        "Date": dates,
        "Open": opens.round(2),
        "High": highs.round(2),
        "Low": lows.round(2),
        "Close": close_path.round(2),
        "Volume": volumes,
        "Amount": amounts,
        "Cash_Dividend": 0.0,
    })


def generate_ohlcv(start_date: str, days: int, start_price: float,
                   trend_type: str,
                   ts_code: str = None, market: str = None) -> pd.DataFrame:
    """从个股历史对应时间段 bootstrap 生成模拟日线 OHLCV。

    1. 加载指定标的的历史行情
    2. 提取 start_date 的日历月份，仅在同月份(±1)的历史窗口中匹配场景特征
    3. 从匹配窗口提取日收益分布，打乱后调整均值，生成模拟价格路径和 OHLCV

    这样生成的模拟K线保留了该股票在对应季节的历史波动特征。
    """
    from core.cache import load as cache_load

    # 回退到全局配置（向后兼容）
    if ts_code is None:
        ts_code = BASE_KWARGS.get("ts_code")
    if market is None:
        market = BASE_KWARGS.get("market")

    target = _SCENARIO_TARGET.get(trend_type)
    if target is None:
        raise ValueError(f"未知趋势类型: {trend_type}")
    target_chg, target_vol, target_up = target
    target_slope = (1 + target_chg / 100) ** (1 / days) - 1

    # 加载该个股的历史数据
    cached = cache_load(ts_code, market)
    if cached is None or cached.empty:
        raise ValueError(f"无法加载 {ts_code} 历史缓存数据")
    close = cached["Close"].values.astype(float)
    dates_full = pd.to_datetime(cached["Date"].values)

    # 提取 start_date 的日历月份，用于季节性匹配
    sim_start_month = pd.Timestamp(start_date).month

    rng = np.random.default_rng({
        "up": 42, "down": 43,
        "volatile_up": 44, "volatile_down": 45,
        "sideways": 46, "v_reversal": 47,
        "inv_v_reversal": 48, "slow_grind_down": 49,
    }[trend_type])

    # 找最相似的历史区间（同日历月份优先）
    s_idx, e_idx, actual_chg, actual_vol, actual_up, score = \
        _find_best_historical_window(close, target_chg, target_vol, target_up,
                                     days, dates_full=dates_full,
                                     start_month=sim_start_month)
    hist_start = pd.Timestamp(dates_full[s_idx]).strftime('%Y-%m-%d')
    hist_end = pd.Timestamp(dates_full[e_idx - 1]).strftime('%Y-%m-%d')
    hist_start_month = pd.Timestamp(dates_full[s_idx]).strftime('%m')

    # bootstrap 生成价格路径
    is_v = trend_type in ("v_reversal", "inv_v_reversal")
    path, daily_ret = _bootstrap_returns(close, s_idx, e_idx, target_slope, days,
                                         rng, v_reverse=is_v)
    # V形先跌后涨，倒V先涨后跌
    if trend_type == "v_reversal":
        half = len(path) // 2
        n_first = half
        mult = (1 - 0.01 * np.arange(n_first, 0, -1) / n_first)[::-1]
        path[:n_first] = path[:n_first] * mult
    elif trend_type == "inv_v_reversal":
        half = len(path) // 2
        n_second = len(path) - half
        mult = (1 - 0.01 * np.arange(n_second) / n_second)
        path[half:] = path[half:] * mult

    close_series = start_price * path

    # 生成 OHLCV
    dates = pd.bdate_range(start=start_date, periods=days)
    df = _wrap_ohlcv(dates, close_series, start_price, rng, ts_code=ts_code)
    df["Date"] = pd.to_datetime(df["Date"])

    sim_chg = (close_series[-1] / close_series[0] - 1) * 100
    month_tag = f"📅同季(始{hist_start_month}月)" if sim_start_month else ""
    print(f"     📊 {ts_code} {trend_type}: ¥{close_series[0]:.1f}→¥{close_series[-1]:.1f} ({sim_chg:+.1f}%)  "
          f"匹配 {hist_start}→{hist_end} ({actual_chg:+.0f}%/σ{actual_vol:.1f}% 得分{score:.2f}) {month_tag}")

    return df.reset_index(drop=True)


def run_one_scenario(df: pd.DataFrame, config):
    """在模拟数据上跑一次回测，返回 (stats, trades, equity, df_sim_indicators)"""
    # 记录模拟起始日期（拼接缓存前）
    sim_start = df["Date"].min()
    start_price = df["Close"].iloc[0]
    end_price = df["Close"].iloc[-1]
    # ---- 拼接缓存历史数据，确保指标计算有足够的预热数据 ----
    from core.cache import load as cache_load
    cached = cache_load(config.ts_code, config.market)
    if cached is not None and not cached.empty:
        hist = cached[cached["Date"] < sim_start].copy()
        if not hist.empty:
            common_cols = df.columns.intersection(hist.columns)
            df = pd.concat([hist[common_cols], df[common_cols]], ignore_index=True)
            df = df.sort_values("Date").reset_index(drop=True)
            start_price = hist["Close"].iloc[-1]
            print(f"   📦 已拼接缓存历史 {len(hist)} 条 (截止 {hist['Date'].max().strftime('%Y-%m-%d')})")

    df = compute_indicators(
        df,
        entry_period=config.entry_period,
        exit_period=config.exit_period,
        atr_period=config.atr_period,
        ema_fast=config.ema_fast,
        ema_slow=config.ema_slow,
        channel_type=config.channel_type,
        bb_period=config.bb_period,
        bb_k=config.bb_k,
        am_ma_period=config.am_ma_period,
        mode=config.mode,
    )
    trades, equity, eq_dates = run_backtest(df, config)
    benchmark_return = round(((end_price - start_price)/start_price)*100,2)
    stats = calc_stats(trades, config.initial_capital, equity, eq_dates,benchmark_return=benchmark_return)

    # 切回模拟期间（去掉缓存历史），用于导出日线数据
    # df_sim = df[df["Date"] >= sim_start].copy()

    return stats, trades, equity, df


# ═══════════════════════════════════════════════════════════════
# 多股票配置 — 每只股票独立生成252天模拟K线，分开保存
# ═══════════════════════════════════════════════════════════════

# 8种场景标签（供外部引用）
SCENARIO_LABELS = [
    ("单边上涨",     "up"),
    ("单边下跌",     "down"),
    ("震荡向上",     "volatile_up"),
    ("震荡向下",     "volatile_down"),
    ("横盘震荡",     "sideways"),
    ("V形反转",      "v_reversal"),
    ("倒V反转",      "inv_v_reversal"),
    ("缓慢阴跌",     "slow_grind_down"),
]

# 所有标的配置（仅生成模拟K线所需字段）
ALL_STOCK_CONFIGS = [
    dict(ts_code="09992.HK", stock_name="泡泡玛特", market="HK"),
    dict(ts_code="02015.HK", stock_name="理想", market="HK"),
    dict(ts_code="03690.HK", stock_name="美团", market="HK"),
    dict(ts_code="01810.HK", stock_name="小米", market="HK"),
    dict(ts_code="06181.HK", stock_name="老铺", market="HK"),
]


def get_last_price_for_stock(ts_code: str, market: str) -> tuple:
    """获取指定标的最近一个交易日的收盘价和日期"""
    from core.cache import load as cache_load
    cached = cache_load(ts_code, market)
    if cached is not None and not cached.empty:
        return float(cached["Close"].iloc[-1]), cached["Date"].iloc[-1]
    raise ValueError(f"{ts_code} 未获取到缓存日线行情")


def generate_stock_simulations(ts_code: str, market: str,
                                trading_days: int = TRADING_DAYS, # 约1年交易日,
                                start_date: str = None) -> dict:
    """为单只股票生成所有 8 种场景的未来模拟K线。

    参数:
        ts_code: 股票代码
        market: 市场
        trading_days: 模拟交易天数 (默认252个交易日≈1年)
        start_date: 模拟起始日期 (默认取最新缓存日期次日)

    返回:
        dict: {scenario_label: DataFrame} 包含8种场景的模拟OHLCV
    """
    from datetime import timedelta

    last_price, last_date = get_last_price_for_stock(ts_code, market)

    if start_date is None:
        start_date = (last_date + timedelta(days=1)).strftime("%Y%m%d")

    print(f"\n{'='*60}")
    print(f"🕯 生成 {ts_code} 未来 {trading_days} 天模拟K线")
    print(f"   最新价: ¥{last_price:.2f}  起始日: {start_date}")
    print(f"{'='*60}")

    sim_data = {}
    for label, trend_type in SCENARIO_LABELS:
        print(f"  🕯 「{label}」...")
        df = generate_ohlcv(start_date, trading_days, last_price, trend_type,
                            ts_code=ts_code, market=market)
        df["scenario"] = label  # 标注场景类型
        sim_data[label] = df

        sp, ep = df["Close"].iloc[0], df["Close"].iloc[-1]
        rets = df["Close"].pct_change().dropna()
        print(f"     价格 {sp:.2f}→{ep:.2f} ({(ep/sp-1)*100:+.2f}%)  "
              f"最低¥{df['Low'].min():.2f} 最高¥{df['High'].max():.2f}  "
              f"日波动σ={rets.std()*100:.2f}%")

    return sim_data


def save_simulations(sim_data: dict, ts_code: str ):
    """将单只股票的模拟K线分开保存为 Parquet 和 Excel。

    目录结构:
        result/simulated_klines/{ts_code}/
        ├── {ts_code}_simulated_252d.parquet    # 全部8场景合并
        ├── {ts_code}_{scenario_label}.parquet  # 单场景
        └── {ts_code}_simulated_252d.xlsx       # Excel（多sheet）
    """

    # ── 合并全部场景保存为单个 Parquet ──
    all_dfs = []
    for label, df in sim_data.items():
        all_dfs.append(df)
    combined = pd.concat(all_dfs, ignore_index=True)
    combined_path = os.path.join(stock_dir, f"{ts_code}_simulated_252d.parquet")
    combined.to_parquet(combined_path, index=False)
    print(f"📁 合并 Parquet: {combined_path} ({len(combined)} 行)")

    # ── 单场景 Parquet ──
    for label, df in sim_data.items():
        safe_label = label.replace("/", "_")
        scene_path = os.path.join(stock_dir, f"{ts_code}_{safe_label}.parquet")
        df.to_parquet(scene_path, index=False)

    # ── Excel（多sheet） ──
    excel_path = os.path.join(stock_dir, f"{ts_code}_simulated_252d.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        # Summary sheet
        summary_rows = []
        for label, df in sim_data.items():
            sp, ep = df["Close"].iloc[0], df["Close"].iloc[-1]
            summary_rows.append({
                "场景": label,
                "起始价": round(sp, 2),
                "结束价": round(ep, 2),
                "涨跌幅%": round((ep/sp-1)*100, 2),
                "最低价": round(df["Low"].min(), 2),
                "最高价": round(df["High"].max(), 2),
                "日波动%": round(df["Close"].pct_change().std()*100, 2),
                "天数": len(df),
            })
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Summary", index=False)

        # 每个场景一个 sheet
        for label, df in sim_data.items():
            safe_label = label.replace("/", "_")[:31]  # Excel sheet名限31字符
            df_export = df.copy()
            df_export["Date"] = df_export["Date"].dt.strftime("%Y-%m-%d")
            df_export.to_excel(writer, sheet_name=safe_label, index=False)

    print(f"📁 Excel: {excel_path}")
    print(f"📁 单场景 Parquet: {stock_dir}/{ts_code}_*.parquet")

    return stock_dir


# ==================== 绘图 ====================

def plot_kline(df: pd.DataFrame, trades: list[dict], label: str, ax):
    """在给定的 ax 上绘制 K 线图 + 交易标记"""
    import matplotlib.dates as mdates

    dates = df["Date"].values
    opens = df["Open"].values
    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values

    # K线颜色：阳线红，阴线绿
    for i in range(len(df)):
        date_num = mdates.date2num(dates[i])
        color = "#e74c3c" if closes[i] >= opens[i] else "#2ecc71"
        o, c, h, l = opens[i], closes[i], highs[i], lows[i]
        ax.plot([date_num, date_num], [l, h], color=color, linewidth=0.8)
        body_h = abs(c - o) or (h - l) * 0.005
        ax.bar(date_num, body_h, 0.6, bottom=min(o, c),
               color=color, edgecolor=color, linewidth=0.3, alpha=0.9)

    # MA 均线
    ma20 = df["Close"].rolling(20).mean()
    ma60 = df["Close"].rolling(60).mean()
    ax.plot(dates, ma20, color="#3498db", linewidth=0.7, alpha=0.7, label="MA20")
    if len(df) >= 60:
        ax.plot(dates, ma60, color="#f39c12", linewidth=0.7, alpha=0.7, label="MA60")

    # 交易标记
    if trades:
        buy_dates = [t["Date"] for t in trades if t["Type"] == "加仓"]
        buy_prices = [t["Price"] for t in trades if t["Type"] == "加仓"]
        sell_dates = [t["Date"] for t in trades if t["Type"] == "减仓"]
        sell_prices = [t["Price"] for t in trades if t["Type"] == "减仓"]
        if buy_dates:
            ax.scatter(buy_dates, buy_prices, marker="^", s=50, color="#e74c3c",
                      edgecolors="white", linewidths=0.3, zorder=5, label=f"加仓({len(buy_dates)})")
        if sell_dates:
            ax.scatter(sell_dates, sell_prices, marker="v", s=50, color="#2ecc71",
                      edgecolors="white", linewidths=0.3, zorder=5, label=f"减仓({len(sell_dates)})")

    sp, ep = closes[0], closes[-1]
    chg = (ep / sp - 1) * 100
    ax.set_title(f"{label}  ¥{sp:.2f}→¥{ep:.2f} ({chg:+.2f}%)", fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=4))
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_ylabel("HKD", fontsize=7)
    ax.legend(fontsize=6, loc="upper left")
    ax.grid(True, alpha=0.3)


def plot_all_klines(sim_data: dict, all_results: dict, output_path: str):
    """多合一 K线图"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️ matplotlib 不可用")
        return

    labels = list(sim_data.keys())
    nrows = (len(labels) + 1) // 2
    fig, axes = plt.subplots(nrows, 2, figsize=(20, 4 * nrows))
    axes = axes.flatten()

    for ax, label in zip(axes, labels):
        df = sim_data[label]
        trades = all_results[label]["trades"]
        plot_kline(df, trades, label, ax)
    # 隐藏多余的子图
    for ax in axes[len(labels):]:
        ax.set_visible(False)

    fig.suptitle("模拟K线 — 多种市场形态   (🔴红=阳线 🟢绿=阴线  ▲加仓  ▼减仓)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"📊 K线图已保存: {output_path}")


def plot_equity_curves(results: dict, output_path: str):
    """多合一 资金曲线图"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️ matplotlib 不可用")
        return

    labels = list(results.keys())
    nrows = (len(labels) + 1) // 2
    fig, axes = plt.subplots(nrows, 2, figsize=(20, 4 * nrows))
    axes = axes.flatten()

    for ax, label in zip(axes, labels):
        equity = results[label]["equity"]
        if not equity:
            ax.set_title(label)
            continue
        ax.plot(equity, linewidth=0.8, color="#2c3e50")
        init = equity[0]
        final = equity[-1]
        ret = (final - init) / init * 100
        ax.axhline(y=init, color="gray", linestyle="--", alpha=0.4)
        ax.set_title(f"{label}  收益率={ret:+.2f}%  最终¥{final:,.0f}", fontsize=10)
        ax.set_ylabel("权益 (¥)", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
    for ax in axes[len(labels):]:
        ax.set_visible(False)

    fig.suptitle("仓位管理策略 — 多种市场形态资金曲线对比", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"📊 资金曲线图已保存: {output_path}")


# ==================== main ====================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="模拟未来K线形态")
    parser.add_argument("--all", action="store_true",
                        help="为全部5只股票生成252天模拟K线，分开保存")
    parser.add_argument("--ts-code", type=str, default="03690.HK",
                        help="股票代码 (默认: 03690.HK)")
    parser.add_argument("--days", type=int, default=252,
                        help="模拟交易天数 (默认252≈1年)")
    parser.add_argument("--no-backtest", action="store_true",
                        help="仅生成K线，跳过回测")
    args = parser.parse_args()

    # ── 批量模式：为全部股票生成模拟K线，分开保存 ──
    if args.all:
        trading_days = args.days
        for cfg in ALL_STOCK_CONFIGS:
            ts_code = cfg["ts_code"]
            market = cfg["market"]
            stock_name = cfg["stock_name"]
            print(f"\n{'#'*60}")
            print(f"# {stock_name} ({ts_code})")
            print(f"{'#'*60}")

            sim_data = generate_stock_simulations(ts_code, market,
                                                  trading_days=trading_days)
            save_simulations(sim_data, ts_code)

        print(f"\n{'='*60}")
        print(f"✅ 全部完成！{len(ALL_STOCK_CONFIGS)} 只股票的模拟K线已保存到 result/simulated_klines/")
        print(f"{'='*60}")
        sys.exit(0)

    # ── 单股票模式 ──
    init_config(args.ts_code)
    setup_cjk_font()

    trading_days = args.days

    # 1. 获取最新价格
    print(f"📥 获取{BASE_KWARGS.get('stock_name')}最新收盘价...")
    LAST_REAL_PRICE, LAST_DATE = get_last_price()
    print(f"   最新收盘价: {LAST_REAL_PRICE:.2f} HKD,最新日期:{LAST_DATE}\n")

    # 2. 生成模拟K线
    sim_data = {}
    for label, trend_type in SCENARIO_LABELS:
        print(f"🕯 生成「{label}」模拟K线 ({trading_days}天) ...")
        start = (LAST_DATE + timedelta(days=1)).strftime("%Y%m%d")
        df = generate_ohlcv(start, trading_days, LAST_REAL_PRICE, trend_type)
        sim_data[label] = df
        sp, ep = df["Close"].iloc[0], df["Close"].iloc[-1]
        lo, hi = df["Low"].min(), df["High"].max()
        rets = df["Close"].pct_change().dropna()
        print(f"   价格 {sp:.2f}→{ep:.2f} ({(ep/sp-1)*100:+.2f}%)  "
              f"最低¥{lo:.2f} 最高¥{hi:.2f}  "
              f"日波动σ={rets.std()*100:.2f}%")

    # 3. 保存模拟K线
    ts_code = BASE_KWARGS["ts_code"]
    stock_dir = os.path.join("result/simulated_klines", ts_code)
    os.makedirs(stock_dir, exist_ok=True)
    save_simulations(sim_data, ts_code)

    if args.no_backtest:
        print("\n✅ 模拟K线生成完成！")
        sys.exit(0)

    # 4. 回测
    print("\n" + "=" * 70)
    print(f"🚀 开始回测 (grid_up={BEST_PARAMS.get('pm_grid_up',0.03):.3f} "
          f"grid_down={BEST_PARAMS.get('pm_grid_down',0.03):.3f})")
    print("=" * 70)

    all_results = {}
    for label, _ in SCENARIO_LABELS:
        print(f"\n--- {label} ---")
        config = BacktestConfig(**{**BASE_KWARGS, **BEST_PARAMS})
        stats, trades, equity, df_sim = run_one_scenario(sim_data[label], config)
        all_results[label] = {"stats": stats, "trades": trades, "equity": equity, "df_sim": df_sim}

        print(f"  交易: {stats['总交易']}次  "
              f"加仓: {stats['盈利次数']+stats['亏损次数']}次  "
              f"收益率: {stats['总收益率 (%)']:+.2f}%  "
              f"盈亏: ¥{stats['总盈亏 (¥)']:+,.2f}  "
              f"回撤: {stats['最大回撤 (%)']:+.2f}%")
        print(f"  最终资金: ¥{stats['最终资金 (¥)']:,.2f}  "
              f"夏普: {stats['夏普比率']:.2f}  "
              f"CAGR: {stats['CAGR (%)']:.2f}%  "
              f"R³: {stats['R立方']:.2f}")

    # 5. 汇总对比
    print("\n" + "=" * 90)
    print("📊 形态汇总对比（按总收益率排序）")
    print("=" * 90)

    sorted_labels = sorted(SCENARIO_LABELS, key=lambda x: all_results[x[0]]["stats"]["总收益率 (%)"], reverse=True)
    print(f"{'形态':<10} {'价格变动':>10} {'总收益率':>10} {'超额收益':>10} {'总盈亏':>14} {'最大回撤':>10} {'交易':>6} {'夏普':>8} {'CAGR':>8}")
    print("-" * 90)
    for label, _ in sorted_labels:
        s = all_results[label]["stats"]
        df = sim_data[label]
        df_sim = all_results[label]["df_sim"]
        cache_df = df_sim[df_sim["Date"] < df["Date"].iloc[0]].copy()
        if not cache_df.empty:
            price_chg = (df["Close"].iloc[-1] / cache_df["Close"].iloc[-1] - 1) * 100
        else:
            price_chg = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100
        print(f"{label:<10} {price_chg:>+9.2f}% {s['总收益率 (%)']:>+9.2f}% {s['超额收益 (%)']:>+9.2f}%"
              f"{s['总盈亏 (¥)']:>+13,.0f}  {s['最大回撤 (%)']:>+9.2f}% "
              f"{s['总交易']:>5}  {s['夏普比率']:>7.2f} {s['CAGR (%)']:>7.2f}%")

    # 6. 交易明细
    print("\n📋 交易明细汇总：")
    for label, _ in SCENARIO_LABELS:
        trades = all_results[label]["trades"]
        if not trades:
            print(f"  {label}: 无交易")
            continue
        tdf = pd.DataFrame(trades)
        buys = tdf[tdf["Type"] == "加仓"]
        sells = tdf[tdf["Type"] == "减仓"]
        print(f"  {label}: 加仓{len(buys)}次 减仓{len(sells)}次  "
              f"买入均价¥{buys['Price'].mean():.2f}" if len(buys) else f"  {label}: 加仓0次 减仓{len(sells)}次  ",
              end="")
        if len(sells):
            print(f"  卖出均价¥{sells['Price'].mean():.2f}")
        else:
            print()

    # 7. 导出回测Excel
    for label, _ in SCENARIO_LABELS:
        r = all_results[label]
        stats = r["stats"]
        trades = r["trades"]
        df_sim = r["df_sim"]

        safe_label = label.replace("/", "_")
        excel_path = f"simulate_{ts_code}_{safe_label}.xlsx"
        with pd.ExcelWriter(os.path.join(config.output_dir, excel_path), engine="openpyxl") as writer:
            summary_data = [
                ["形态", label],
                ["模拟天数", trading_days],
                ["交易次数", stats["总交易"]],
                ["胜率 (%)", stats["胜率 (%)"]],
                ["赔率", stats["赔率"]],
                ["期望收益率", stats["期望收益率"]],
                ["总盈亏 (¥)", stats["总盈亏 (¥)"]],
                ["总收益率 (%)", stats["总收益率 (%)"]],
                ["超额收益 (%)", stats["超额收益 (%)"]],
                ["最大回撤 (%)", stats["最大回撤 (%)"]],
                ["最终资金 (¥)", stats["最终资金 (¥)"]],
                ["夏普比率", stats["夏普比率"]],
                ["CAGR (%)", stats["CAGR (%)"]],
                ["MAR比率", stats["MAR比率"]],
                ["衰落持续期 (天)", stats["衰落持续期 (天)"]],
                ["回归年度回报率 (%)", stats["回归年度回报率 (%)"]],
                ["R平方", stats["R平方"]],
                ["R立方", stats["R立方"]],
                ["稳健夏普比率", stats["稳健夏普比率"]],
            ]
            pd.DataFrame(summary_data, columns=["指标", "数值"]).to_excel(
                writer, sheet_name="Summary", index=False)

            df_export = df_sim.copy()
            df_export["Date"] = df_export["Date"].dt.strftime("%Y-%m-%d")
            df_export.to_excel(writer, sheet_name="日线原始数据", index=False)

            if trades:
                tdf = pd.DataFrame(trades)
                pm_cols = ["Date", "Type", "Price", "Shares", "Amount",
                           "Position_After", "Capital_After", "Equity_After",
                           "Reference_Price", "Reason"]
                tdf[pm_cols].to_excel(writer, sheet_name="仓位管理明细", index=False)

        print(f"📁 {label}: {excel_path}")

    # 8. 绘图
    plot_all_klines(sim_data, all_results, os.path.join(stock_dir, f"simulate_market_klines_{ts_code}.png"))
    plot_equity_curves(all_results, os.path.join(stock_dir, f"simulate_market_equity_{ts_code}.png"))

    print("\n✅ 完成！")
