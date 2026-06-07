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


def _check_long_exit(row, stop_loss_price):
    """检查做多出场条件。返回 (exit_price, reason)；无信号返回 (None, '')。"""
    if row["Close"] < stop_loss_price:
        return row["Close"], "Hard Stop Loss (2×ATR)"
    if row["Close"] < row["LL_Exit"]:
        return row["Close"], "Trend Reversal (10d Low)"
    return None, ""


def _check_both_exit(row, pos, stop_loss_price):
    """检查双向出场条件。返回 (exit_price, reason)；无信号返回 (None, '')。"""
    if pos == "long":
        if row["Close"] < stop_loss_price:
            return row["Close"], "Hard Stop Loss (2×ATR)"
        if row["Close"] < row["LL_Exit"]:
            return row["Close"], "Trend Reversal (10d Low)"
    else:  # short
        if row["Close"] > stop_loss_price:
            return row["Close"], "Hard Stop Loss (2×ATR)"
        if row["Close"] > row["HL_Exit"]:
            return row["Close"], "Trend Reversal (10d High)"
    return None, ""


# ==================== 输出格式化 ====================

def _print_header(label, initial_capital, risk_percent):
    print(f"\n🔍 开始运行 {label} (带动态仓位管理)...")
    print(f"   初始资金：{initial_capital:,.2f} | 单笔风险：{risk_percent:.1f}%")


def _print_entry(label, price, shares, cost, capital, stop_loss):
    print(f"  🟢 [{label}] 做多入场：{price:.2f} | "
          f"股数：{shares} | 投入：{cost:.2f} | "
          f"剩余资金：{capital:.2f} | 止损：{stop_loss:.2f}")


def _print_short_entry(label, price, shares, income, capital, stop_loss):
    print(f"  🟢 [{label}] 做空入场：{price:.2f} | "
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

def _run_long_only(df: pd.DataFrame, config) -> tuple[list[dict], list[float]]:
    label = "System 1"
    ps = config.per_shares
    _print_header(label, config.initial_capital, config.risk_percent)

    trades = []
    capital = config.initial_capital
    equity_curve = [config.initial_capital]

    pos = None
    entry_price = 0.0
    stop_loss_price = 0.0
    stop_dist = 0.0
    current_shares = 0
    entry_date = None

    start_index = config.entry_period + 1

    for i in range(start_index, len(df)):
        row = df.iloc[i]
        current_date = row["Date"]

        if pos is None:
            stop_dist, shares = _prep_entry(row, capital, config, ps)
            if shares <= 0:
                continue

            if row["Close"] > row["HH_Entry"]:
                cost = row["Close"] * shares * ps
                if not _can_afford(cost, capital):
                    continue
                pos = "long"
                entry_price = row["Close"]
                stop_loss_price = entry_price - stop_dist
                current_shares = shares
                entry_date = current_date
                capital -= cost
                _print_entry(label, row["Close"], shares * ps, cost, capital, stop_loss_price)

        else:
            exit_price, exit_reason = _check_long_exit(row, stop_loss_price)
            if exit_reason:
                cost_basis = entry_price * current_shares * ps
                pnl = (exit_price - entry_price) * current_shares * ps
                capital += exit_price * current_shares * ps

                trades.append(_build_trade(
                    entry_date, current_date, "Long",
                    entry_price, exit_price, current_shares,
                    cost_basis, pnl, capital, exit_reason,
                    stop_dist=stop_dist,
                ))
                equity_curve.append(capital)
                _print_exit(label, pnl, capital, exit_reason)
                pos = None
                current_shares = 0
                entry_price = 0.0
                stop_loss_price = 0.0
                entry_date = None

    # 强制平仓
    if pos == "long":
        last_price = df.iloc[-1]["Close"]
        last_date = df.iloc[-1]["Date"]
        cost_basis = entry_price * current_shares * ps
        pnl = (last_price - entry_price) * current_shares * ps
        capital += last_price * current_shares * ps

        trades.append(_build_trade(
            entry_date, last_date, "Long",
            entry_price, last_price, current_shares,
            cost_basis, pnl, capital, "强制平仓",
            stop_dist=stop_dist,
        ))
        _print_force_close(label, pnl, capital)

    return trades, equity_curve


# ==================== 加仓做多（金字塔） ====================

def _run_long_with_pyramid(df: pd.DataFrame, config) -> tuple[list[dict], list[float]]:
    label = "System 1"
    ps = config.per_shares
    _print_header(label, config.initial_capital, config.risk_percent)

    trades = []
    capital = config.initial_capital
    equity_curve = [config.initial_capital]

    pos = None
    entry_price = 0.0
    entry_atr = 0.0
    stop_loss_price = 0.0
    stop_dist = 0.0
    current_shares = 0
    base_shares = 0
    max_shares = 0
    total_cost = 0.0
    entry_date = None
    more_info = []

    start_index = config.entry_period + 1

    for i in range(start_index, len(df)):
        row = df.iloc[i]
        current_date = row["Date"]

        if pos is None:
            stop_dist, shares = _prep_entry(row, capital, config, ps)
            if shares <= 0:
                continue
            base_shares = shares
            skip_offset = 1 if config.skip_first_add else 0
            max_shares = shares * (config.max_pyramid - skip_offset)

            if row["Close"] > row["HH_Entry"]:
                cost = row["Close"] * shares * ps
                if not _can_afford(cost, capital):
                    continue
                pos = "long"
                entry_price = row["Close"]
                entry_atr = row["ATR"]
                stop_loss_price = entry_price - stop_dist
                current_shares = shares
                total_cost = cost
                entry_date = current_date
                capital -= cost
                _print_entry(label, row["Close"], shares * ps, cost, capital, stop_loss_price)

        else:
            exit_price, exit_reason = _check_long_exit(row, stop_loss_price)

            # 加仓（仅在未触发出场信号时检查，以入场 ATR 为基准）
            if not exit_reason and entry_atr > 0:
                units_held = current_shares // base_shares + skip_offset
                if row["Close"] > (entry_price + units_held * 0.5 * entry_atr):
                    if current_shares < max_shares:
                        add_price = row["Close"]
                        stop_loss_price = add_price - config.n_value * entry_atr
                        add_cost = add_price * base_shares * ps
                        if not _can_afford(add_cost, capital):
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
                        _print_pyramid_add(label, add_price, base_shares * ps,
                                          add_cost, capital, stop_loss_price)

            if exit_reason:
                sell_proceeds = exit_price * current_shares * ps
                pnl = sell_proceeds - total_cost
                capital += sell_proceeds

                trades.append(_build_trade(
                    entry_date, current_date, "Long",
                    entry_price, exit_price, current_shares,
                    total_cost, pnl, capital, exit_reason,
                    stop_dist=stop_dist, more_info=more_info,
                ))
                equity_curve.append(capital)
                _print_exit(label, pnl, capital, exit_reason)
                pos = None
                current_shares = 0
                base_shares = 0
                max_shares = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_loss_price = 0.0
                total_cost = 0.0
                entry_date = None
                more_info = []

    # 强制平仓
    if pos == "long":
        last_price = df.iloc[-1]["Close"]
        last_date = df.iloc[-1]["Date"]
        sell_proceeds = last_price * current_shares * ps
        pnl = sell_proceeds - total_cost
        capital += sell_proceeds

        trades.append(_build_trade(
            entry_date, last_date, "Long",
            entry_price, last_price, current_shares,
            total_cost, pnl, capital, "强制平仓",
            stop_dist=stop_dist, more_info=more_info,
        ))
        _print_force_close(label, pnl, capital)

    return trades, equity_curve


# ==================== 多空双向 ====================

def _run_long_short(df: pd.DataFrame, config) -> tuple[list[dict], list[float]]:
    label = "System 1"
    PS = 100  # 多空脚本硬编码，不使用 config.per_shares
    _print_header(label, config.initial_capital, config.risk_percent)

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
            stop_dist, shares = _prep_entry(row, capital, config, PS)
            if shares <= 0:
                continue

            entry_price = row["Close"]

            if row["Close"] > row["HH_Entry"]:
                cost = entry_price * shares * PS
                if cost > capital * 0.95:
                    continue
                pos = "long"
                stop_loss_price = entry_price - stop_dist
                current_shares = shares
                entry_date = current_date
                capital -= cost
                _print_entry(label, entry_price, shares * PS, cost, capital, stop_loss_price)

            elif row["Close"] < row["LL_Entry"]:
                income = entry_price * shares * PS
                pos = "short"
                stop_loss_price = entry_price + stop_dist
                current_shares = -shares
                entry_date = current_date
                capital += income
                _print_short_entry(label, entry_price, shares * PS, income, capital, stop_loss_price)

        else:
            exit_price, exit_reason = _check_both_exit(row, pos, stop_loss_price)
            if exit_reason:
                pnl = (exit_price - entry_price) * current_shares * PS
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
                equity_curve.append(capital)
                _print_exit(label, pnl, capital, exit_reason)
                pos = None
                current_shares = 0
                entry_price = 0.0
                stop_loss_price = 0.0
                entry_date = None

    # 强制平仓
    if pos is not None:
        last_price = df.iloc[-1]["Close"]
        last_date = df.iloc[-1]["Date"]
        pnl = (last_price - entry_price) * current_shares * PS
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
        _print_force_close(label, pnl, capital)

    return trades, equity_curve
