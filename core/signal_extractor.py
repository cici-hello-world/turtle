"""仓位信号提取模块

用法：
    from core.signal_extractor import extract_signal, format_signal
    signal = extract_signal(config)
"""

from dataclasses import dataclass
from fileinput import close

import pandas as pd

from core.config import BacktestConfig
from core.data import fetch_data
from core.indicators import compute_indicators
from core.engine import run_backtest, _dmi_trend_area, _calc_effective_grid


@dataclass
class DailySignal:
    ts_code: str
    stock_name: str
    signal_date: str

    # 持仓状态
    current_lots: int
    current_shares: int
    reference_price: float
    last_trade_date: str
    last_type: str
    last_shares: int
    last_price: float

    # 今日 DMI
    adx: float | None
    adxr: float | None
    plus_di: float | None
    minus_di: float | None
    adx_up: bool
    dmi_area: str          # A / B / C / D / "-"
    is_bull: bool

    # 有效网格（DMI 自适应后）
    eff_grid_down: float
    eff_grid_up: float
    eff_reduce: float

    # 触发价 & 数量
    add_trigger: float
    add_lots: int
    reduce_trigger: float
    reduce_lots: int

    # 加仓是否被 DMI 拦截（short方向 A/B 区）
    dmi_blocked: bool
    today_close: float

    # 超额收益（策略收益 - 个股买入持有收益）
    excess_return: float | None


def extract_signal(config: BacktestConfig) -> DailySignal | None:
    """加载数据 → 计算指标 → 回测 → 提取下一交易日仓位信号。

    优先读缓存，不触发网络请求（调用前需确保缓存已更新）。
    返回 None 表示数据不足或异常。
    """
    try:
        end_date = pd.Timestamp.today().strftime("%Y%m%d")
        if pd.Timestamp.now().hour < 18:
            # 晚上6点之后才拉取当前的，否则值拉取到前一天
            end_date = (pd.Timestamp.today() - pd.DateOffset(days=1)).strftime("%Y%m%d")
        df = fetch_data(
            ts_code=config.ts_code,
            stock_name=config.stock_name,
            market=config.market,
            start_date=config.start_date,
            end_date=end_date,
            exchange=config.exchange,
            data_source=config.data_source,
            use_cache=True,
        )
    except Exception as e:
        print(f"[signal] {config.stock_name} 数据加载失败: {e}")
        return None

    if df is None or df.empty:
        print(f"[signal] {config.stock_name} 无数据")
        return None

    # 2. 计算指标
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
        pm_bb_period=config.pm_bb_period,
        pm_bb_k=config.pm_bb_k,
        pm_bb_width_lookback=config.pm_bb_width_lookback,
    )

    # 3. 运行回测，取最后持仓状态
    try:
        trades, equity_curve, _ = run_backtest(df, config)
    except Exception as e:
        print(f"[signal] {config.stock_name} 回测失败: {e}")
        return None

    # 4. 今日 DMI 状态（最后一行）
    today = df.iloc[-1]
    signal_date = str(today["Date"])[:10]
    today_close=today["Close"]

    adx = today.get("ADX14")
    adxr = today.get("ADXR")
    plus_di = today.get("+DI")
    minus_di = today.get("-DI")
    adx_up = bool(today.get("ADX_UP", False))
    dmi_bl = df.attrs.get("dmi_baseline")  # 个股 DMI 基线
    dmi_area = _dmi_trend_area(today, config, baseline=dmi_bl) or "-"
    is_bull = (plus_di or 0) > (minus_di or 0)

    # 5. 有效网格
    eff_down, eff_up, eff_reduce = _calc_effective_grid(today, config, dmi_bl)

    # 6. 持仓状态
    if trades:
        last = trades[-1]
        # 如果最后一笔交易是强制平仓，往前取最近一笔非强平的真实交易
        if last.get("Reason") == "强制平仓":
            real_trades = [t for t in trades if t.get("Reason") != "强制平仓"]
            last = real_trades[-1] if real_trades else None
    else:
        last = None

    if last is not None:
        current_lots = int(last.get("Position_After", 0))
        reference_price = float(last.get("Reference_Price", 0))
        last_trade_date = str(last.get("Date", ""))[:10]
        last_equity = float(last.get("Equity_After", 0))
        last_type = last.get("Type", "无")
        last_shares = int(last.get("Shares", 0))
        last_price = reference_price
    else:
        current_lots = config.pm_initial_shares
        last_trade_date = "-"
        last_type = "-"
        last_shares = "-"
        last_price = "-"
        start_date = pd.to_datetime(config.pm_start_date)
        mask = pd.to_datetime(df["Date"]) >= start_date
        reference_price = float(df.loc[mask, "Close"].iloc[0]) if mask.any() else float(today["Close"])
        last_equity = config.initial_capital + current_lots * reference_price * config.per_shares
    # 参考价重置：空仓时逐日扫描收盘价，防止牛市踏空
    if config.pm_adaptive and current_lots == 0:
        if last is not None:
            start_date = pd.to_datetime(last_trade_date)
        else:
            start_date = pd.to_datetime(config.pm_start_date)
        mask = (pd.to_datetime(df["Date"]) > start_date) & (pd.to_datetime(df["Date"]) <= pd.to_datetime(signal_date))
        for _, row in df[mask].iterrows():
            day_close = row["Close"]
            if day_close > reference_price * (1 + eff_up):
                reference_price = day_close
    current_shares = current_lots * config.per_shares

    # 6.5 超额收益：策略收益 - 个股同期买入持有收益
    try:
        if equity_curve and len(equity_curve) >= 2:
            strategy_return = (equity_curve[-1] / equity_curve[0] - 1) * 100
        else:
            strategy_return = 0.0

        if config.pm_start_date:
            start_date = pd.to_datetime(config.pm_start_date)
            mask = pd.to_datetime(df["Date"]) >= start_date
            if mask.any():
                start_close = float(df.loc[mask, "Close"].iloc[0])
                bh_return = (today_close / start_close - 1) * 100
                excess_return = round(strategy_return - bh_return, 2)
            else:
                excess_return = None
        else:
            excess_return = None
    except Exception:
        excess_return = None

    # 7. 触发价
    add_trigger = round(reference_price * (1 - eff_down), 2)
    reduce_trigger = round(reference_price * (1 + eff_up), 2)

    # 8. 加仓量 & 减仓量
    if current_lots == 0:
        add_lots = _calc_lots(last_equity, config.pm_add_pct, add_trigger, config.per_shares)
        reduce_lots = 0
    else:
        add_lots = _calc_lots(last_equity, config.pm_add_pct, add_trigger, config.per_shares)
        reduce_lots = min(
            max(1, int(last_equity * eff_reduce / (reduce_trigger * config.per_shares) + 0.5)),
            current_lots,
        )

    # 9. DMI 加仓拦截：short方向 A/B 区 = 强下跌趋势，不放行加仓
    area_short = _dmi_trend_area(today, config, "short", dmi_bl)
    dmi_blocked = area_short in ("A", "B")

    return DailySignal(
        ts_code=config.ts_code,
        stock_name=config.stock_name,
        signal_date=signal_date,
        current_lots=current_lots,
        current_shares=current_shares,
        reference_price=reference_price,
        last_trade_date=last_trade_date,
        adx=round(float(adx), 1) if adx is not None and not pd.isna(adx) else None,
        adxr=round(float(adxr), 1) if adxr is not None and not pd.isna(adxr) else None,
        plus_di=round(float(plus_di), 1) if plus_di is not None and not pd.isna(plus_di) else None,
        minus_di=round(float(minus_di), 1) if minus_di is not None and not pd.isna(minus_di) else None,
        adx_up=adx_up,
        dmi_area=dmi_area,
        is_bull=is_bull,
        eff_grid_down=eff_down,
        eff_grid_up=eff_up,
        eff_reduce=eff_reduce,
        add_trigger=add_trigger,
        add_lots=add_lots,
        reduce_trigger=reduce_trigger,
        reduce_lots=reduce_lots,
        dmi_blocked=dmi_blocked,
        today_close=today_close,
        last_type=last_type,
        last_shares=last_shares,
        last_price=last_price,
        excess_return=excess_return,
    )


def _calc_lots(equity: float, pct: float, price: float, per_shares: int) -> int:
    """按权益百分比和每手股数计算手数（向下取整）"""
    value = equity * pct
    lots = int(value / (price * per_shares))
    return max(0, lots)


def format_signal(s: DailySignal) -> str:
    """将信号格式化为通知消息文本"""
    # DMI 状态
    adx_up_str = "↑" if s.adx_up else "↓"
    dmi_detail = (
        f"DMI: {s.dmi_area}区 {'多头' if s.is_bull else '空头'}  "
        f"ADX{s.adx or '?'}{adx_up_str}  +DI{s.plus_di or '?'} −DI{s.minus_di or '?'}"
    )
    trade_detail=None

    # # 网格描述
    # if s.dmi_area == "A" and s.is_bull:
    #     grid_desc = "宽·让利润跑"
    # elif s.dmi_area in ("B", "C") and s.is_bull:
    #     grid_desc = "中·适度止盈"
    # elif not s.is_bull:
    #     grid_desc = "窄·防守"
    # elif s.dmi_area == "D":
    #     grid_desc = "窄·震荡"
    # else:
    #     grid_desc = "默认"

    # 持仓
    if s.current_lots > 0:
        pos = f"持仓 {s.current_lots}手({s.current_shares}股)  参考价 {s.reference_price:.2f}"
    else:
        pos = f"空仓  参考价 {s.reference_price:.2f}"
    if s.last_trade_date and s.last_trade_date != "-":
        trade_detail = f"上次交易 {s.last_trade_date}  {s.last_type}价格{s.last_price} 数量{s.last_shares}手"

    lines = [
        f"**{s.stock_name}** {s.ts_code} --{s.signal_date}收盘价 {s.today_close}",
        f"  {pos}",
        f"  {trade_detail if trade_detail else "无交易"}",
        f"  {dmi_detail}",
        f"  网格: 下{s.eff_grid_down*100:.1f}% 上{s.eff_grid_up*100:.1f}%",
    ]

    # 超额收益
    if s.excess_return is not None:
        sign = "+" if s.excess_return >= 0 else ""
        lines.append(f"  超额收益: {sign}{s.excess_return:.2f}%")

    # 触发信号
    dmi_tag = " ⚠️DMI拦截" if s.dmi_blocked else ""
    if s.current_lots > 0:
        lines.append(f"  🔴 加仓 ≤{s.add_trigger}  +{s.add_lots}手{dmi_tag}")
        lines.append(f"  🟢 减仓 ≥{s.reduce_trigger}  −{s.reduce_lots}手")
    else:
        lines.append(f"  🔴 建仓 ≤{s.add_trigger}  +{s.add_lots}手{dmi_tag}")
        lines.append(f"  🟢 空仓")

    return "\n".join(lines)
