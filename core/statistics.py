"""交易统计计算"""

import pandas as pd
import numpy as np


def calc_stats(trades: list[dict], initial_capital: float, equity_curve: list[float]) -> dict:
    """基于交易列表和资金曲线计算统计指标"""
    if not trades:
        return {
            "总交易": 0,
            "胜率 (%)": 0,
            "赔率": 0,
            "期望收益率": 0,
            "盈利次数": 0,
            "亏损次数": 0,
            "总盈亏 (¥)": 0,
            "总收益率 (%)": 0,
            "最大回撤 (%)": 0,
            "最终资金 (¥)": initial_capital,
            "夏普比率": 0,
            "平均盈亏 (¥)": 0,
        }

    df_trades = pd.DataFrame(trades)

    # 排除强制平仓
    df_trades = df_trades[df_trades["Reason"] != "强制平仓"]

    if len(df_trades) == 0:
        return {
            "总交易": 0,
            "胜率 (%)": 0,
            "赔率": 0,
            "期望收益率": 0,
            "盈利次数": 0,
            "亏损次数": 0,
            "总盈亏 (¥)": 0,
            "总收益率 (%)": 0,
            "最大回撤 (%)": 0,
            "最终资金 (¥)": initial_capital,
            "夏普比率": 0,
            "平均盈亏 (¥)": 0,
        }

    wins = df_trades[df_trades["PnL_%"] > 0]
    losses = df_trades[df_trades["PnL_%"] < 0]

    final_capital = df_trades.iloc[-1]["Capital_After"]
    total_trades = len(df_trades)
    win_count = len(wins)
    loss_count = len(losses)
    win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0

    total_pnl = final_capital - initial_capital
    total_return = (total_pnl / initial_capital) * 100

    # 最大回撤（基于 equity_curve）
    equity_series = pd.Series(equity_curve)
    cummax = equity_series.cummax()
    drawdown = (equity_series - cummax) / cummax
    max_dd = drawdown.min() * 100

    # 平均盈亏
    avg_win = wins["PnL_¥"].mean() if win_count > 0 else 0
    avg_loss = abs(losses["PnL_¥"].mean()) if loss_count > 0 else 0
    win_loss_ratio = (avg_win / avg_loss) if avg_loss > 0 else 0

    # 期望值
    expectation = (win_rate / 100 * win_loss_ratio) - (100 - win_rate) / 100

    # 夏普比率
    if len(equity_curve) > 1:
        returns = equity_series.pct_change().dropna()
        if len(returns) > 0 and returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            sharpe = 0
    else:
        sharpe = 0

    return {
        "总交易": total_trades,
        "胜率 (%)": round(win_rate, 2),
        "赔率": win_loss_ratio,
        "期望收益率": expectation,
        "盈利次数": win_count,
        "亏损次数": loss_count,
        "总盈亏 (¥)": round(total_pnl, 2),
        "总收益率 (%)": round(total_return, 2),
        "最大回撤 (%)": round(max_dd, 2),
        "最终资金 (¥)": round(final_capital, 2),
        "夏普比率": round(sharpe, 2),
        "平均盈亏 (¥)": round((avg_win + avg_loss) / 2, 2),
    }
