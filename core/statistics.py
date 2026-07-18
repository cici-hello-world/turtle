"""交易统计计算"""

import pandas as pd
import numpy as np
from itertools import groupby

from keyring.backends import null


def _build_equity_dates(trades: list[dict], equity_curve: list[float]) -> pd.Series | None:
    """为资金曲线每个点构造对应日期，失败返回 None"""
    if not trades or ("Entry Date" not in trades[0] and "Date" not in trades[0]):
        return None
    try:
        # 仓位管理策略使用 "Date" 字段
        if "Date" in trades[0]:
            dates = [pd.to_datetime(trades[0]["Date"])]
            for t in trades:
                dates.append(pd.to_datetime(t["Date"]))
            # 对齐 equity_curve：如果末尾多一个最终权益点，补最后一个日期
            while len(dates) < len(equity_curve):
                dates.append(dates[-1])
            return pd.Series(dates)

        first_entry = pd.to_datetime(trades[0]["Entry Date"])
        dates = [first_entry]  # equity_curve[0] 对应起始日
        for t in trades:
            if t.get("Reason") == "强制平仓":
                continue
            dates.append(pd.to_datetime(t["Exit Date"]))
        return pd.Series(dates)
    except Exception:
        return None


def _compute_total_years(trades: list[dict], equity_curve: list[float],
                         equity_dates: pd.Series | None) -> float:
    """计算回测总年数"""
    if equity_dates is not None:
        return (equity_dates.iloc[-1] - equity_dates.iloc[0]).days / 365.25
    # 无日期时返回0表示未发生交易
    return 0


def _compute_cagr(initial_capital: float, final_capital: float,
                  total_years: float) -> float:
    """年化复合增长率 CAGR"""
    if total_years > 0 and initial_capital > 0 and final_capital > 0:
        return (final_capital / initial_capital) ** (1 / total_years) - 1
    return 0


def _compute_drawdown_duration(equity_series: pd.Series,
                               equity_dates: pd.Series | None) -> int:
    """最长衰落持续期（天）；无日期时返回交易事件数"""
    cummax = equity_series.cummax()
    is_underwater = equity_series < cummax

    # 找最长连续水下区间
    max_duration = 0
    for underwater, group in groupby(enumerate(is_underwater), key=lambda x: x[1]):
        if not underwater:
            continue
        indices = [x[0] for x in group]
        start_idx, end_idx = indices[0], indices[-1]
        if equity_dates is not None:
            dur = int((equity_dates.iloc[end_idx] - equity_dates.iloc[start_idx]).days)
        else:
            dur = end_idx - start_idx
        max_duration = max(max_duration, dur)
    return max_duration


def _compute_regression_metrics(equity_series: pd.Series,
                                equity_dates: pd.Series | None) -> tuple[float, float]:
    """回归年度回报率 与 R²"""
    if len(equity_series) < 3:
        return 0, 0

    log_equity = np.log(equity_series.values.astype(float))

    if equity_dates is not None:
        # 时间轴以年为单位
        t = np.array([(d - equity_dates.iloc[0]).days / 365.25
                      for d in equity_dates])
    else:
        return 0, 0

    # 线性回归 log(equity) = a + b * t
    A = np.vstack([np.ones_like(t), t]).T
    # NumPy 2.x 要求 b 为二维，reshape 后 squeeze 回一维
    coeffs, residuals, rank, s = np.linalg.lstsq(A, log_equity[:, np.newaxis], rcond=None)
    a, b = coeffs.flatten()  # b 是连续复利年化斜率

    # 年化简单回报率RAR
    reg_annual_return = np.exp(b) - 1

    # R²
    log_equity_pred = a + b * t
    ss_res = np.sum((log_equity - log_equity_pred) ** 2)
    ss_tot = np.sum((log_equity - np.mean(log_equity)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    return reg_annual_return, r_squared


def _compute_r_cubed(equity_series: pd.Series, reg_annual_return: float,
                     equity_dates: pd.Series | None) -> float:
    """R立方 (R-Cubed): 稳健风险回报比率

    R³ = RAR / LAMD
    LAMD = 平均最大衰落 × (平均回撤天数 / 365)

    分子 RAR 用回归斜率而非端点值，分母用5次最大回撤的均值而非单次极值，
    整个指标对起止日期和数据的轻微扰动不敏感。
    """
    # 最少60个交易日（约3个月），避免短周期R³极端膨胀
    if len(equity_series) < 60:
        return 0.0
    # 收益接近零时返回0，避免无意义的极大/极小值
    if abs(reg_annual_return) < 1e-10:
        return 0.0

    cummax = equity_series.cummax()
    drawdown_pct = (equity_series - cummax) / cummax  # ≤0，负值表示回撤

    # 识别所有水下区间
    is_underwater = drawdown_pct < -1e-6
    if not is_underwater.any():
        return 0

    periods = []
    for underwater, group in groupby(enumerate(is_underwater), key=lambda x: x[1]):
        if not underwater:
            continue
        indices = [x[0] for x in group]
        start_idx, end_idx = indices[0], indices[-1]

        max_dd = abs(drawdown_pct.iloc[start_idx:end_idx + 1].min())

        if equity_dates is not None:
            dur = int((equity_dates.iloc[end_idx] - equity_dates.iloc[start_idx]).days)
        else:
            dur = end_idx - start_idx

        periods.append({"max_dd": max_dd, "duration": max(dur, 1)})

    if not periods:
        return 0

    # 按回撤幅度降序，取前5
    periods.sort(key=lambda x: x["max_dd"], reverse=True)
    top5 = periods[:5]

    avg_max_dd = float(np.mean([p["max_dd"] for p in top5]))
    avg_duration = float(np.mean([p["duration"] for p in top5]))

    if avg_max_dd <= 0 or avg_duration <= 0:
        return 0

    lamd = avg_max_dd * (avg_duration / 365)
    if lamd <= 0:
        return 0

    return reg_annual_return / lamd


def _compute_robust_sharpe(equity_series: pd.Series, reg_annual_return: float,
                          equity_dates: pd.Series | None) -> float:
    """稳健夏普比率 = 回归年度回报率 / 年度化的月度回报标准差"""
    if len(equity_series) < 2 or abs(reg_annual_return) < 1e-10:
        return 0.0
    if equity_dates is None or len(equity_dates) != len(equity_series):
        return 0
    # 构建 DatetimeIndex 后按月末重采样
    eq = pd.Series(equity_series.values, index=pd.DatetimeIndex(equity_dates))
    monthly_eq = eq.resample("ME").last().dropna()
    if len(monthly_eq) < 3:
        return 0
    monthly_returns = monthly_eq.pct_change().dropna()
    monthly_std = monthly_returns.std()
    if monthly_std <= 0:
        return 0
    annualized_vol = monthly_std * np.sqrt(12)
    return reg_annual_return / annualized_vol


def calc_stats(trades: list[dict], initial_capital: float, equity_curve: list[float],
               equity_dates: list | None = None,
               benchmark_return: float | None = None) -> dict:
    """基于交易列表和资金曲线计算统计指标

    equity_dates:     与 equity_curve 一一对应的日期序列（日频）。
                      传入时直接使用，不传则从 trades 推断（兼容旧调用）。
    benchmark_return: 同期买入持有收益率（%），用于计算超额收益。
                      不传则超额收益为 NaN。
    """
    empty_result = {
        "总交易": 0,
        "donchian交易":0,
        "donchian盈利次数":0,
        "donchian强制平仓":0,
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
        "CAGR (%)": 0,
        "MAR比率": 0,
        "超额收益 (%)": float("nan"),
        "衰落持续期 (天)": 0,
        "回归年度回报率 (%)": 0,
        "R平方": 0,
        "稳健夏普比率": 0,
        "R立方": 0,
        "last_entry_date":null,
        "last_exit_date":null

    }

    if not trades:
        return empty_result

    df_trades = pd.DataFrame(trades)

    # 仓位管理策略：使用 "Date" 字段，用法不同
    is_pm = "Position_After" in df_trades.columns
    if is_pm:
        last_entry_date = df_trades["Date"].iloc[-1]
        last_exit_date = df_trades["Date"].iloc[-1]
        # 排除强制平仓
        df_trades_no_fc = df_trades[df_trades["Reason"] != "强制平仓"]
        # 最终资金用 equity_curve 的最后一个值
        final_capital = equity_curve[-1] if equity_curve else initial_capital
    else:
        last_entry_date = df_trades["Entry Date"].iloc[-1]
        last_exit_date = df_trades["Exit Date"].iloc[-1]
        # 排除强制平仓
        df_trades_no_fc = df_trades[df_trades["Reason"] != "强制平仓"]
        if len(df_trades_no_fc) == 0:
            final_capital = equity_curve[-1] if equity_curve else initial_capital
        else:
            final_capital = df_trades_no_fc.iloc[-1]["Capital_After"]
    df_trades = df_trades_no_fc

    if len(df_trades) == 0:
        pm_init = equity_curve[0] if (is_pm and equity_curve) else initial_capital
        result = empty_result.copy()
        result["最终资金 (¥)"] = round(final_capital, 2)
        result["总盈亏 (¥)"] = round(final_capital - pm_init, 2)
        result["总收益率 (%)"] = round((final_capital - pm_init) / pm_init * 100, 2) if pm_init > 0 else 0
        result["last_entry_date"] = last_entry_date
        result["last_exit_date"] = last_exit_date
        return result

    wins = df_trades[df_trades["PnL_%"] > 0]
    losses = df_trades[df_trades["PnL_%"] < 0]
    donchian_mask = df_trades["Reason"].str.contains("Donchian买入", na=False)
    donchian_trades = int(donchian_mask.sum())
    # Donchian 盈利次数：Donchian买入行 + 下一行（减仓）的 Amount 之和 > 0
    donchian_idx = df_trades.index[donchian_mask]
    donchian_wins = 0
    donchian_forced = 0
    for idx in donchian_idx:
        # 找到该索引在 df_trades 中的位置
        pos = df_trades.index.get_loc(idx)
        if pos + 1 < len(df_trades):
            entry_amount = df_trades.iloc[pos]["Amount"]
            exit_amount = df_trades.iloc[pos + 1]["Amount"]
            if entry_amount + exit_amount > 0:
                donchian_wins += 1
            if df_trades.iloc[pos + 1]["Reason"] == "强制平仓":
                donchian_forced += 1
    total_trades = len(df_trades)
    win_count = len(wins)
    loss_count = len(losses)
    win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0

    # 仓位管理策略：初始权益 = equity_curve[0]（含持仓市值），而非纯现金
    pm_initial = equity_curve[0] if (is_pm and equity_curve) else initial_capital
    total_pnl = final_capital - pm_initial
    total_return = (total_pnl / pm_initial) * 100 if pm_initial > 0 else 0

    # 最大衰落（%）
    equity_series = pd.Series(equity_curve)
    cummax = equity_series.cummax()
    drawdown = (equity_series - cummax) / cummax
    max_dd = drawdown.min() * 100

    # 赔率
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

    # ——— 日期序列：优先用引擎传入的日频日期，否则从 trades 推断 ———
    if equity_dates is not None and len(equity_dates) == len(equity_curve):
        eq_dates = pd.Series(pd.to_datetime(equity_dates))
    else:
        eq_dates = _build_equity_dates(trades, equity_curve)
    total_years = _compute_total_years(trades, equity_curve, eq_dates)

    # CAGR
    cagr = _compute_cagr(pm_initial, final_capital, total_years)

    # MAR 比率 cagr为负数时失效
    if abs(max_dd) > 0.01 and cagr>0:
        mar_ratio = (cagr * 100) / abs(max_dd)
    else:
        mar_ratio = 0

    # 衰落持续期
    dd_duration = _compute_drawdown_duration(equity_series, eq_dates)

    # 回归年度回报率 & R²
    reg_return, r_squared = _compute_regression_metrics(equity_series, eq_dates)

    # 稳健夏普比率
    robust_sharpe = _compute_robust_sharpe(equity_series, reg_return, eq_dates)

    # R立方
    r_cubed = _compute_r_cubed(equity_series, reg_return, eq_dates)

    # 超额收益：策略总收益率 - 同期买入持有收益率
    if benchmark_return is not None:
        excess_return = round(total_return - benchmark_return, 2)
    else:
        excess_return = float("nan")

    return {
        "总交易": total_trades,
        "donchian交易": donchian_trades,
        "donchian盈利次数": donchian_wins,
        "donchian强制平仓": donchian_forced,
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
        "CAGR (%)": round(cagr * 100, 2),
        "MAR比率": round(mar_ratio, 2),
        "衰落持续期 (天)": dd_duration,
        "回归年度回报率 (%)": round(reg_return * 100, 2),
        "R平方": round(r_squared, 4),
        "稳健夏普比率": round(robust_sharpe, 2),
        "R立方": round(r_cubed, 2),
        "超额收益 (%)": excess_return,
        "last_entry_date": last_entry_date,
        "last_exit_date": last_exit_date
    }
