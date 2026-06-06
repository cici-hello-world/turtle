"""海龟策略回测引擎

支持三种模式:
- "long":      不加仓做多
- "long_more": 加仓做多（金字塔）
- "both":      多空双向
"""

import numpy as np
import pandas as pd


def run_backtest(df: pd.DataFrame, config) -> tuple[list[dict], list[float]]:
    """统一回测入口，返回 (trades, equity_curve)"""
    if config.mode == "long":
        return _run_long_only(df, config)
    elif config.mode == "long_more":
        return _run_long_with_pyramid(df, config)
    elif config.mode == "both":
        return _run_long_short(df, config)
    else:
        raise ValueError(f"未知的策略模式: {config.mode}")


# ==================== 共享工具函数 ====================

def _calc_shares(capital: float, risk_per_trade: float, stop_dist: float,
                 per_shares: int) -> int:
    """计算入场股数（手数），基于风险资金 / 止损距离"""
    return int(capital * risk_per_trade / (stop_dist * per_shares))


def _can_afford(cost: float, capital: float, ratio: float = 0.8) -> bool:
    """检查资金是否足够"""
    return cost <= capital * ratio


# ==================== 不加仓做多 ====================

def _run_long_only(df: pd.DataFrame, config) -> tuple[list[dict], list[float]]:
    print(f"\n🔍 开始运行 System 1 (带动态仓位管理)...")
    print(f"   初始资金：{config.initial_capital:,.2f} | 单笔风险：{config.risk_percent:.1f}%")

    trades = []
    capital = config.initial_capital
    equity_curve = [config.initial_capital]

    pos = None          # None | 'long'
    entry_price = 0.0
    stop_loss_price = 0.0
    stop_dist = 0.0     # 记录入场时的 2*ATR
    current_shares = 0
    entry_date = None

    start_index = config.entry_period + 1

    for i in range(start_index, len(df)):
        row = df.iloc[i]
        current_date = row["Date"]

        if pos is None:
            if pd.isna(row["ATR"]) or row["ATR"] == 0:
                continue

            atr = row["ATR"]
            stop_dist = config.n_value * atr
            shares = _calc_shares(capital, config.risk_per_trade, stop_dist, config.per_shares)
            if shares <= 0:
                continue

            # 做多入场
            if row["Close"] > row["HH_Entry"]:
                cost = row["Close"] * shares * config.per_shares
                if not _can_afford(cost, capital):
                    continue

                pos = "long"
                entry_price = row["Close"]
                stop_loss_price = entry_price - stop_dist
                current_shares = shares
                entry_date = current_date
                capital -= cost

                print(f"  🟢 [System 1] 做多入场：{row['Close']:.2f} | "
                      f"股数：{shares * config.per_shares} | 投入：{cost:.2f} | "
                      f"剩余资金：{capital:.2f} | 止损：{stop_loss_price:.2f}")

        else:
            # 持仓中，检查出场
            is_stopped = False
            exit_price = 0.0
            exit_reason = ""

            # 硬止损
            if row["Close"] < stop_loss_price:
                is_stopped = True
                exit_price = row["Close"]
                exit_reason = "Hard Stop Loss (2×ATR)"
            # 趋势反转
            elif row["Close"] < row["LL_Exit"]:
                is_stopped = True
                exit_price = row["Close"]
                exit_reason = "Trend Reversal (10d Low)"

            if is_stopped:
                cost_basis = entry_price * current_shares * config.per_shares
                pnl_amount = (exit_price - entry_price) * current_shares * config.per_shares
                sell_proceeds = exit_price * current_shares * config.per_shares
                capital += sell_proceeds
                pnl_pct = pnl_amount / cost_basis

                trades.append({
                    "Entry Date": entry_date,
                    "Exit Date": current_date,
                    "Type": "Long",
                    "Entry Price": entry_price,
                    "2ATR": stop_dist,
                    "Exit Price": exit_price,
                    "Shares": current_shares,
                    "Cost_¥": cost_basis,
                    "PnL_¥": pnl_amount,
                    "PnL_%": pnl_pct,
                    "Capital_After": capital,
                    "Reason": exit_reason,
                })

                equity_curve.append(capital)

                status = "🟢 盈利" if pnl_amount > 0 else "🔴 亏损"
                print(f"  {status} [System 1] 平仓：{exit_reason} | "
                      f"盈亏：{pnl_amount:,.2f}¥ | 剩余资金：{capital:,.2f}")

                pos = None
                current_shares = 0
                entry_price = 0.0
                stop_loss_price = 0.0
                entry_date = None

    # 强制平仓
    if pos == "long":
        last_price = df.iloc[-1]["Close"]
        last_date = df.iloc[-1]["Date"]
        cost_basis = entry_price * current_shares * config.per_shares
        pnl_amount = (last_price - entry_price) * current_shares * config.per_shares
        sell_proceeds = last_price * current_shares * config.per_shares
        capital += sell_proceeds

        trades.append({
            "Entry Date": entry_date,
            "Exit Date": last_date,
            "Type": "Long",
            "Entry Price": entry_price,
            "2ATR": stop_dist,
            "Exit Price": last_price,
            "Shares": current_shares,
            "Cost_¥": cost_basis,
            "PnL_¥": pnl_amount,
            "PnL_%": pnl_amount / cost_basis,
            "Capital_After": capital,
            "Reason": "强制平仓",
        })
        print(f"  ⚠️ [System 1] 强制平仓：{pnl_amount:,.2f}¥ | 最终资金：{capital:,.2f}")

    return trades, equity_curve


# ==================== 加仓做多（金字塔） ====================

def _run_long_with_pyramid(df: pd.DataFrame, config) -> tuple[list[dict], list[float]]:
    print(f"\n🔍 开始运行 System 1 (带动态仓位管理)...")
    print(f"   初始资金：{config.initial_capital:,.2f} | 单笔风险：{config.risk_percent:.1f}%")

    trades = []
    capital = config.initial_capital
    equity_curve = [config.initial_capital]

    pos = None
    entry_price = 0.0
    stop_loss_price = 0.0
    stop_dist = 0.0
    current_shares = 0
    base_shares = 0           # 单次加仓股数（固定单位）
    max_shares = 0            # 最大持仓股数
    total_cost = 0.0          # 累计成本（含加仓）
    entry_date = None
    more_info = []

    start_index = config.entry_period + 1

    for i in range(start_index, len(df)):
        row = df.iloc[i]
        current_date = row["Date"]

        if pos is None:
            if pd.isna(row["ATR"]) or row["ATR"] == 0:
                continue

            atr = row["ATR"]
            stop_dist = config.n_value * atr
            shares = _calc_shares(capital, config.risk_per_trade, stop_dist, config.per_shares)
            if shares <= 0:
                continue

            base_shares = shares
            max_shares = shares * config.max_pyramid

            # 做多入场
            if row["Close"] > row["HH_Entry"]:
                cost = row["Close"] * shares * config.per_shares
                if not _can_afford(cost, capital):
                    continue

                pos = "long"
                entry_price = row["Close"]
                stop_loss_price = entry_price - stop_dist
                current_shares = shares
                total_cost = cost
                entry_date = current_date
                capital -= cost

                print(f"  🟢 [System 1] 做多入场：{row['Close']:.2f} | "
                      f"股数：{shares * config.per_shares} | 投入：{cost:.2f} | "
                      f"剩余资金：{capital:.2f} | 止损：{stop_loss_price:.2f}")

        else:
            is_stopped = False
            exit_price = 0.0
            exit_reason = ""

            atr = row["ATR"]

            # 1. 硬止损
            if row["Close"] < stop_loss_price:
                is_stopped = True
                exit_price = row["Close"]
                exit_reason = "Hard Stop Loss (2×ATR)"
            # 2. 趋势反转
            elif row["Close"] < row["LL_Exit"]:
                is_stopped = True
                exit_price = row["Close"]
                exit_reason = "Trend Reversal (10d Low)"
            # 3. 加仓条件
            elif not pd.isna(atr) and row["Close"] > (entry_price + (current_shares / base_shares) * 0.5 * atr):
                if current_shares < max_shares:
                    more_price = row["Close"]
                    stop_loss_price = more_price - config.n_value * atr
                    more_cost = more_price * base_shares * config.per_shares
                    if more_cost > capital * 0.8:
                        continue
                    current_shares += base_shares
                    total_cost += more_cost
                    capital -= more_cost
                    more_info.append({
                        "More Price": more_price,
                        "More Date": current_date.strftime("%Y%m%d"),
                    })
                    print(f"  🟢 [System 1] 做多加仓：{more_price:.2f} | "
                          f"股数：{base_shares * config.per_shares} | 投入：{more_cost:.2f} | "
                          f"剩余资金：{capital:.2f} | 止损上移至：{stop_loss_price:.2f}")

            if is_stopped:
                sell_proceeds = exit_price * current_shares * config.per_shares
                pnl_amount = sell_proceeds - total_cost
                capital += sell_proceeds
                pnl_pct = pnl_amount / total_cost

                trades.append({
                    "Entry Date": entry_date,
                    "Exit Date": current_date,
                    "Type": "Long",
                    "Entry Price": entry_price,
                    "More Info": more_info,
                    "2ATR": stop_dist,
                    "Exit Price": exit_price,
                    "Shares": current_shares,
                    "Cost_¥": total_cost,
                    "PnL_¥": pnl_amount,
                    "PnL_%": pnl_pct,
                    "Capital_After": capital,
                    "Reason": exit_reason,
                })

                equity_curve.append(capital)

                status = "🟢 盈利" if pnl_amount > 0 else "🔴 亏损"
                print(f"  {status} [System 1] 平仓：{exit_reason} | "
                      f"盈亏：{pnl_amount:,.2f}¥ | 剩余资金：{capital:,.2f}")

                pos = None
                current_shares = 0
                base_shares = 0
                max_shares = 0
                entry_price = 0.0
                stop_loss_price = 0.0
                total_cost = 0.0
                entry_date = None
                more_info = []

    # 强制平仓
    if pos == "long":
        last_price = df.iloc[-1]["Close"]
        last_date = df.iloc[-1]["Date"]
        cost_basis = entry_price * current_shares * config.per_shares
        pnl_amount = (last_price - entry_price) * current_shares * config.per_shares
        sell_proceeds = last_price * current_shares * config.per_shares
        capital += sell_proceeds

        trades.append({
            "Entry Date": entry_date,
            "Exit Date": last_date,
            "Type": "Long",
            "Entry Price": entry_price,
            "More Info": more_info,
            "2ATR": stop_dist,
            "Exit Price": last_price,
            "Shares": current_shares,
            "Cost_¥": cost_basis,
            "PnL_¥": pnl_amount,
            "PnL_%": pnl_amount / cost_basis,
            "Capital_After": capital,
            "Reason": "强制平仓",
        })
        print(f"  ⚠️ [System 1] 强制平仓：{pnl_amount:,.2f}¥ | 最终资金：{capital:,.2f}")

    return trades, equity_curve


# ==================== 多空双向 ====================

def _run_long_short(df: pd.DataFrame, config) -> tuple[list[dict], list[float]]:
    print(f"\n🔍 开始运行 System 1 (带动态仓位管理)...")
    print(f"   初始资金：{config.initial_capital:,.2f} | 单笔风险：{config.risk_percent:.1f}%")

    trades = []
    capital = config.initial_capital
    equity_curve = [config.initial_capital]

    pos = None          # None | 'long' | 'short'
    entry_price = 0.0
    stop_loss_price = 0.0
    current_shares = 0  # long 为正, short 为负
    entry_date = None

    start_index = config.entry_period + 1

    for i in range(start_index, len(df)):
        row = df.iloc[i]
        current_date = row["Date"]

        if pos is None:
            if pd.isna(row["ATR"]) or row["ATR"] == 0:
                continue

            atr = row["ATR"]
            stop_dist = config.n_value * atr
            # 多空脚本使用硬编码 100（不是 PER_SHARES），保持兼容
            shares = _calc_shares(capital, config.risk_per_trade, stop_dist, 100)
            if shares <= 0:
                continue

            entry_price = row["Close"]

            # 做多信号
            if row["Close"] > row["HH_Entry"]:
                cost = row["Close"] * shares * 100
                if cost > capital * 0.95:
                    continue
                pos = "long"
                stop_loss_price = entry_price - stop_dist
                current_shares = shares
                entry_date = current_date
                capital -= cost
                print(f"  🟢 [System 1] 做多入场：{row['Close']:.2f} | "
                      f"股数：{shares * 100} | 投入：{cost:.2f} | "
                      f"剩余资金：{capital:.2f} | 止损：{stop_loss_price:.2f}")

            # 做空信号
            elif row["Close"] < row["LL_Entry"]:
                income = row["Close"] * shares * 100
                pos = "short"
                stop_loss_price = entry_price + stop_dist
                current_shares = -shares
                entry_date = current_date
                capital += income
                print(f"  🟢 [System 1] 做空入场：{row['Close']:.2f} | "
                      f"股数：{shares * 100} | 收入：{income:.2f} | "
                      f"剩余资金：{capital:.2f} | 止损：{stop_loss_price:.2f}")

        else:
            is_stopped = False
            exit_price = 0.0
            exit_reason = ""

            if pos == "long":
                if row["Close"] < stop_loss_price:
                    is_stopped = True
                    exit_price = row["Close"]
                    exit_reason = "Hard Stop Loss (2×ATR)"
                elif row["Close"] < row["LL_Exit"]:
                    is_stopped = True
                    exit_price = row["Close"]
                    exit_reason = "Trend Reversal (10d Low)"
            elif pos == "short":
                if row["Close"] > stop_loss_price:
                    is_stopped = True
                    exit_price = row["Close"]
                    exit_reason = "Hard Stop Loss (2×ATR)"
                elif row["Close"] > row["HL_Exit"]:
                    is_stopped = True
                    exit_price = row["Close"]
                    exit_reason = "Trend Reversal (10d High)"

            if is_stopped:
                pnl_amount = (exit_price - entry_price) * current_shares * 100
                sell_proceeds = exit_price * current_shares * 100
                capital += sell_proceeds

                if pos == "long":
                    cost_basis = entry_price * current_shares * 100
                    pnl_pct = pnl_amount / cost_basis
                else:
                    cost_basis = abs(exit_price * current_shares * 100)
                    pnl_pct = pnl_amount / cost_basis

                trades.append({
                    "Entry Date": entry_date,
                    "Exit Date": current_date,
                    "Type": pos.capitalize(),
                    "Entry Price": entry_price,
                    "Exit Price": exit_price,
                    "Shares": current_shares,
                    "Cost_¥": cost_basis,
                    "PnL_¥": pnl_amount,
                    "PnL_%": pnl_pct,
                    "Capital_After": capital,
                    "Reason": exit_reason,
                })

                equity_curve.append(capital)

                status = "🟢 盈利" if pnl_amount > 0 else "🔴 亏损"
                print(f"  {status} [System 1] 平仓：{exit_reason} | "
                      f"盈亏：{pnl_amount:,.2f}¥ | 剩余资金：{capital:,.2f}")

                pos = None
                current_shares = 0
                entry_price = 0.0
                stop_loss_price = 0.0
                entry_date = None

    # 强制平仓
    if pos is not None:
        last_price = df.iloc[-1]["Close"]
        last_date = df.iloc[-1]["Date"]
        pnl_amount = (last_price - entry_price) * current_shares * 100
        sell_proceeds = last_price * current_shares * 100

        if pos == "long":
            cost_basis = entry_price * current_shares * 100
            pnl_pct = pnl_amount / cost_basis
            capital += sell_proceeds
        else:
            cost_basis = abs(exit_price * current_shares * 100)
            pnl_pct = pnl_amount / cost_basis
            capital -= sell_proceeds

        trades.append({
            "Entry Date": entry_date,
            "Exit Date": last_date,
            "Type": pos.capitalize(),
            "Entry Price": entry_price,
            "Exit Price": last_price,
            "Shares": current_shares,
            "Cost_¥": cost_basis,
            "PnL_¥": pnl_amount,
            "PnL_%": pnl_pct,
            "Capital_After": capital,
            "Reason": "强制平仓",
        })
        print(f"  ⚠️ [System 1] 强制平仓：{pnl_amount:,.2f}¥ | 最终资金：{capital:,.2f}")

    return trades, equity_curve
