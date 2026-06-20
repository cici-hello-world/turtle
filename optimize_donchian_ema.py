#!/usr/bin/env python3
"""唐奇安 EMA 趋势过滤器参数优化 —— 基于流动性 TOP10 加仓做多回测

功能：
  1. 筛选近五年日均成交额最高的 10 只股票 (港股/ A股)
  2. 对每只股票运行唐奇安通道突破 + 金字塔加仓回测
  3. 对比不同 EMA 快/慢线参数组合的效果
  4. 输出综合排名与最优参数建议

用法：
  python optimize_donchian_ema.py          # 默认港股
  python optimize_donchian_ema.py A        # A股

依赖：
  tushare, akshare (仅港股分红补权), pandas, numpy
  环境变量 TUSHARE_TOKEN 需要已配置
"""

import os
import re
import sys
import pandas as pd
import numpy as np
from itertools import product
import warnings
import tushare as ts

from core import BacktestConfig, compute_indicators, calc_stats
from core.data import _fetch_hk_share, _fetch_a_share
from core.engine import _run_long_with_pyramid

warnings.filterwarnings("ignore")

# ============================================================
# 参数配置
# ============================================================
# 市场切换: 环境变量 TURTLE_MARKET=A 或命令行传入 "A"
_raw = sys.argv[1].upper() if len(sys.argv) > 1 else os.environ.get("TURTLE_MARKET", "HK")
MARKET = _raw if _raw in ("HK", "A") else "HK"

START_DATE = "20180618"
END_DATE = "20260618"
INITIAL_CAPITAL = 200000.0
RISK_PERCENT = 1.0
N_VALUE = 2.0
PER_SHARES = 100
MAX_PYRAMID = 5
SKIP_FIRST_ADD = False

ENTRY_PERIOD = 20
EXIT_PERIOD = 10
ATR_PERIOD = 14

FAST_PERIODS = [20, 25, 30, 40, 50]
SLOW_PERIODS = [50, 60, 80, 100, 150, 200, 250, 300, 350]

# 输出路径
_suffix = "a" if MARKET == "A" else "hk"
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), f"donchian_param_analysis_pyramid_{_suffix}.csv")


# ============================================================
# A股: adj_factor 前复权
# ============================================================
if MARKET == "A":
    import akshare as _dummy  # noqa (仅港股需要)

    def fetch_liquidity(pro, start: str, end: str):
        """A股: 沪深300+中证500成分股 → 按日均成交额筛选>1亿"""
        # 获取沪深300成分股
        idx300 = pro.index_weight(index_code='000300.SH', start_date='20260601', end_date='20260615')
        codes_300 = idx300['con_code'].unique().tolist()
        # 获取中证500成分股
        idx500 = pro.index_weight(index_code='000905.SH', start_date='20260601', end_date='20260615')
        codes_500 = idx500['con_code'].unique().tolist()
        # 合并去重
        candidates = list(set(codes_300 + codes_500))
        rank = []
        for ts_code in candidates:
            # 拉日线
            df = _fetch_a_share(pro, ts_code, start, end)
            avg_amt = df["Amount"].mean() #单位千元
            if avg_amt > 100_000:  # 日均成交额 > 1亿
                rank.append((ts_code, avg_amt, df))
        rank.sort(key=lambda x: x[1], reverse=True)
        print(f"按日均成交额筛选>1亿的股票有{len(rank)}只")
        return rank

else:
    def fetch_liquidity(pro, start: str, end: str):
        """港股: 恒生指数成分股 → 按日均成交额筛选>1亿"""
        # 获取恒生指数成分股

        idx = pro.hk_daily(trade_date='20260615')
        candidates = idx[idx["amount"] > 1e8]["ts_code"].tolist()
        rank = []
        for ts_code in candidates:
            try:
                df = _fetch_hk_share(pro, ts_code, start, end)
            except ValueError:
                continue
            avg_amt = df["Amount"].mean() #单位港元
            if avg_amt > 100_000_000:  # 日均成交额 > 1亿 (HKD)
                rank.append((ts_code, avg_amt, df))
        rank.sort(key=lambda x: x[1], reverse=True)
        print(f"按日均成交额筛选>1亿的股票有{len(rank)}只")
        return rank

# ---- else 块结束，以下是共享函数 ----

# ============================================================
# 唐奇安指标计算
# ============================================================
# def compute_donchian_indicators(
#     df: pd.DataFrame,
#     entry_period: int = ENTRY_PERIOD,
#     exit_period: int = EXIT_PERIOD,
#     atr_period: int = ATR_PERIOD,
# ) -> pd.DataFrame:
#     """计算唐奇安通道与 ATR 指标"""
#     df = df.copy()
#     df = df.sort_values("trade_date").reset_index(drop=True)
#
#     df["TR"] = np.maximum(
#         df["high"] - df["low"],
#         np.maximum(
#             abs(df["high"] - df["close"].shift(1)),
#             abs(df["low"] - df["close"].shift(1)),
#         ),
#     )
#     df["ATR"] = df["TR"].rolling(atr_period, min_periods=atr_period).mean()
#     df["HH_Entry"] = df["high"].rolling(entry_period).max().shift(1)
#     df["LL_Entry"] = df["low"].rolling(entry_period).min().shift(1)
#     df["LL_Exit"] = df["low"].rolling(exit_period).min().shift(1)
#     return df


# def add_ema_filter(df: pd.DataFrame, ema_fast: int, ema_slow: int) -> pd.DataFrame:
#     """添加 EMA 趋势过滤器"""
#     df = df.copy()
#     if ema_fast > 0:
#         df["EMA_Fast"] = df["close"].ewm(span=ema_fast, adjust=False).mean()
#     if ema_slow > 0:
#         df["EMA_Slow"] = df["close"].ewm(span=ema_slow, adjust=False).mean()
#     return df


# ============================================================
# 加仓做多回测（对齐 engine.py 的 _run_long_with_pyramid）
# ============================================================
# def backtest_donchian_pyramid(
#     df: pd.DataFrame,
#     ema_fast: int = 0,
#     ema_slow: int = 0,
#     entry_period: int = ENTRY_PERIOD,
#     exit_period: int = EXIT_PERIOD,
#     atr_period: int = ATR_PERIOD,
#     n_value: float = N_VALUE,
#     risk_percent: float = RISK_PERCENT / 100.0,
#     initial_capital: float = INITIAL_CAPITAL,
#     per_shares: int = PER_SHARES,
#     max_pyramid: int = MAX_PYRAMID,
#     skip_first_add: bool = SKIP_FIRST_ADD,
# ) -> tuple[list[dict], list[float]]:
#     """唐奇安通道突破 + 金字塔加仓回测。返回 (trades, equity_curve)。"""
#     df = compute_donchian_indicators(df, entry_period, exit_period, atr_period)
#     df = add_ema_filter(df, ema_fast, ema_slow)
#
#     trades = []
#     capital = initial_capital
#     equity_curve = [initial_capital]
#
#     pos = None
#     entry_price = 0.0
#     entry_atr = 0.0
#     stop_loss_price = 0.0
#     stop_dist = 0.0
#     current_shares = 0
#     base_shares = 0
#     max_shares = 0
#     total_cost = 0.0
#     entry_date = None
#     more_info = []
#
#     start_idx = max(entry_period, atr_period, max(ema_slow, 1)) + 2
#
#     for i in range(start_idx, len(df)):
#         row = df.iloc[i]
#         current_date = row["trade_date"]
#
#         if pos is None:
#             # ---- 等待入场 ----
#             atr = row["ATR"]
#             if pd.isna(atr) or atr == 0:
#                 continue
#
#             stop_dist = n_value * atr
#             risk_amount = capital * risk_percent
#             shares = int(risk_amount / (stop_dist * per_shares))
#             if shares <= 0:
#                 continue
#             base_shares = shares
#             skip_offset = 1 if skip_first_add else 0
#             max_shares = shares * (max_pyramid - skip_offset)
#
#             ema_ok = True
#             if ema_fast > 0 and ema_slow > 0:
#                 fast_val = row.get("EMA_Fast")
#                 slow_val = row.get("EMA_Slow")
#                 if pd.notna(fast_val) and pd.notna(slow_val):
#                     ema_ok = fast_val > slow_val
#
#             if row["close"] > row["HH_Entry"] and ema_ok:
#                 cost = row["close"] * shares * per_shares
#                 if cost > capital * 0.8:
#                     continue
#                 pos = "long"
#                 entry_price = row["close"]
#                 entry_atr = row["ATR"]
#                 stop_loss_price = entry_price - stop_dist
#                 current_shares = shares
#                 total_cost = cost
#                 entry_date = current_date
#                 capital -= cost
#
#         else:
#             # ---- 检查出场 ----
#             exit_price = None
#             exit_reason = ""
#
#             if row["close"] < stop_loss_price:
#                 exit_price = row["close"]
#                 exit_reason = "Hard Stop Loss (2×ATR)"
#             elif row["close"] < row["LL_Exit"]:
#                 exit_price = row["close"]
#                 exit_reason = "Trend Reversal (10d Low)"
#
#             # ---- 金字塔加仓 ----
#             if not exit_reason and entry_atr > 0:
#                 skip_offset = 1 if skip_first_add else 0
#                 units_held = current_shares // base_shares + skip_offset
#                 if row["close"] > (entry_price + units_held * 0.5 * entry_atr):
#                     if current_shares < max_shares:
#                         add_price = row["close"]
#                         stop_loss_price = add_price - n_value * entry_atr
#                         add_cost = add_price * base_shares * per_shares
#                         if add_cost <= capital * 0.8:
#                             current_shares += base_shares
#                             total_cost += add_cost
#                             capital -= add_cost
#                             more_info.append({
#                                 "add_price": add_price,
#                                 "add_date": current_date.strftime("%Y%m%d"),
#                             })
#
#             # ---- 出场处理 ----
#             if exit_reason:
#                 sell_proceeds = exit_price * current_shares * per_shares
#                 pnl = sell_proceeds - total_cost
#                 capital += sell_proceeds
#
#                 trades.append({
#                     "entry_date": entry_date,
#                     "exit_date": current_date,
#                     "entry_price": entry_price,
#                     "exit_price": exit_price,
#                     "shares": current_shares,
#                     "cost": total_cost,
#                     "pnl": pnl,
#                     "pnl_pct": pnl / total_cost if total_cost else 0,
#                     "reason": exit_reason,
#                     "adds": len(more_info),
#                 })
#                 equity_curve.append(capital)
#
#                 pos = None
#                 entry_price = 0.0
#                 entry_atr = 0.0
#                 stop_loss_price = 0.0
#                 current_shares = 0
#                 base_shares = 0
#                 max_shares = 0
#                 total_cost = 0.0
#                 entry_date = None
#                 more_info = []
#
#     # ---- 强制平仓 ----
#     if pos is not None:
#         last_price = df.iloc[-1]["close"]
#         last_date = df.iloc[-1]["trade_date"]
#         sell_proceeds = last_price * current_shares * per_shares
#         pnl = sell_proceeds - total_cost
#         capital += sell_proceeds
#
#         trades.append({
#             "entry_date": entry_date,
#             "exit_date": last_date,
#             "entry_price": entry_price,
#             "exit_price": last_price,
#             "shares": current_shares,
#             "cost": total_cost,
#             "pnl": pnl,
#             "pnl_pct": pnl / total_cost if total_cost else 0,
#             "reason": "强制平仓",
#             "adds": len(more_info),
#         })
#         equity_curve.append(capital)
#
#     return trades, equity_curve


# ============================================================
# 交易评估
# ============================================================
def evaluate(trades: list[dict], equity_curve: list[float],
             initial_capital: float = INITIAL_CAPITAL) -> dict:
    """评估交易结果，返回关键指标"""
    if not trades:
        return {
            "total_trades": 0, "win_rate": 0, "total_pnl": 0,
            "total_return_pct": 0, "avg_win": 0, "avg_loss": 0,
            "profit_factor": 0, "max_drawdown_pct": 0,
            "avg_trade_pnl": 0, "final_capital": initial_capital,
            "total_adds": 0,
        }

    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]

    total_pnl = sum(t["pnl"] for t in trades)
    win_rate = len(wins) / len(trades) if trades else 0
    avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0
    avg_loss = np.mean([t["pnl"] for t in losses]) if losses else 0
    total_wins_sum = sum(t["pnl"] for t in wins) if wins else 0
    total_losses_sum = abs(sum(t["pnl"] for t in losses)) if losses else 0
    profit_factor = (total_wins_sum / total_losses_sum) if total_losses_sum > 0 else (
        float("inf") if total_wins_sum > 0 else 0)
    total_adds = sum(t.get("adds", 0) for t in trades)

    # 最大回撤
    peak = equity_curve[0]
    max_dd = 0.0
    for v in equity_curve:
        if v > peak:
            peak = v
        dd = (peak - v) / peak
        if dd > max_dd:
            max_dd = dd

    final_capital = equity_curve[-1]

    return {
        "total_trades": len(trades),
        "win_rate": round(win_rate * 100, 1),
        "total_pnl": round(total_pnl, 2),
        "total_return_pct": round((final_capital / initial_capital - 1) * 100, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "avg_trade_pnl": round(total_pnl / len(trades), 2) if trades else 0,
        "final_capital": round(final_capital, 2),
        "total_adds": total_adds,
    }




# ============================================================
# 参数扫描 & 综合评分
# ============================================================
def run_parameter_scan(top10_stocks,name_map) -> pd.DataFrame:
    """对 TOP10 股票扫描所有 EMA 参数组合，返回综合评分 DataFrame"""
    all_results = {}  # ts_code -> [(fast, slow, metrics), ...]

    for rank, (ts_code, avg_amt, df_raw) in enumerate(top10_stocks, 1):
        name = name_map.get(ts_code, "")
        results = []
        # 无过滤器基准
        config = BacktestConfig(
            ts_code=ts_code,
            stock_name=name,
            channel_type="donchian",
            ema_fast=0,
            ema_slow=0,
        )
        df = compute_indicators(
            df_raw,
            entry_period=config.entry_period,
            exit_period=config.exit_period,
            atr_period=config.atr_period,
            ema_fast=config.ema_fast,
            ema_slow=config.ema_slow,
            channel_type=config.channel_type,
            bb_period=config.bb_period,
            bb_k=config.bb_k,
        )
        trades, equity = _run_long_with_pyramid(df,config)
        baseline = calc_stats(trades, config.initial_capital, equity)
        baseline["ema_fast"] = 0
        baseline["ema_slow"] = 0
        results.append((0, 0, baseline))

        # 参数扫描
        for fast, slow in product(FAST_PERIODS, SLOW_PERIODS):
            if slow <= fast:
                continue
            config.ema_fast = fast
            config.ema_slow = slow
            df = compute_indicators(
                df_raw,
                entry_period=config.entry_period,
                exit_period=config.exit_period,
                atr_period=config.atr_period,
                ema_fast=config.ema_fast,
                ema_slow=config.ema_slow,
                channel_type=config.channel_type,
                bb_period=config.bb_period,
                bb_k=config.bb_k,
            )
            trades, eq = _run_long_with_pyramid(df, config)
            metrics = calc_stats(trades, config.initial_capital, equity)
            metrics["ema_fast"] = fast
            metrics["ema_slow"] = slow
            results.append((fast, slow, metrics))

        all_results[ts_code] = results

        best = max(results, key=lambda x: x[2]["R立方"])#最优参数评价依据？
        print(f"#{rank} {ts_code} (日均成交 {avg_amt:,.0f})")
        print(f"  基准(无过滤):  R立方 {baseline['R立方']}, 回归年度回报率 {baseline['回归年度回报率 (%)']}%"
              f"期望收益率 {baseline['期望收益率']:>5.1f}%, 稳健夏普比率 {baseline['稳健夏普比率']:>3}, "
              f"回撤 {baseline['最大回撤 (%)']:>5.1f}%, 盈亏比 {baseline['赔率']:>5.2f}"
             )
        print(f"  最佳(总回报): EMA({best[0]}/{best[1]}) → "
              f"R立方 {best[2]['R立方']:>7.1f}%, 回归年度回报率 {best[2]['回归年度回报率 (%)']}%"
              f"期望收益率 {best[2]['期望收益率']:>5.1f}%, 稳健夏普比率 {best[2]['稳健夏普比率']:>3}, "
              f"回撤 {best[2]['最大回撤 (%)']:>5.1f}%, 盈亏比 {best[2]['赔率']:>5.2f}"
              )

    # ---- 跨股票汇总 ----
    param_scores = {}
    for fast, slow in product(FAST_PERIODS, SLOW_PERIODS):
        if slow <= fast:
            continue
        param_scores[(fast, slow)] = {
            "r_dd": [], "returns": [],
            "expect_rates": [], "sharpe": [],
            "max_dds": [], "profit_factors": [],
            "r_dd_vs_baseline": [], "return_vs_baseline": [], "dd_vs_baseline": [],
        }
    param_scores[(0, 0)] = {
        "r_dd": [],"returns": [],
        "expect_rates": [], "sharpe": [],
        "max_dds": [], "profit_factors": [],
        "r_dd_vs_baseline": [],"return_vs_baseline": [], "dd_vs_baseline": [],
    }

    for ts_code, results in all_results.items():
        baseline_return = None
        baseline_dd = None
        for fast, slow, metrics in results:
            if fast == 0 and slow == 0:
                baseline_return = metrics["回归年度回报率 (%)"]
                baseline_dd = metrics["最大回撤 (%)"]
                baseline_r_dd = metrics["R立方"]
                param_scores[(0, 0)]["returns"].append(metrics["回归年度回报率 (%)"])
                param_scores[(0, 0)]["expect_rates"].append(metrics["期望收益率"])
                param_scores[(0, 0)]["profit_factors"].append(metrics["赔率"])
                param_scores[(0, 0)]["max_dds"].append(metrics["最大回撤 (%)"])
                param_scores[(0, 0)]["sharpe"].append(metrics["稳健夏普比率"])
                param_scores[(0, 0)]["r_dd"].append(metrics["R立方"])
            else:
                param_scores[(fast, slow)]["returns"].append(metrics["回归年度回报率 (%)"])
                param_scores[(fast, slow)]["expect_rates"].append(metrics["期望收益率"])
                param_scores[(fast, slow)]["profit_factors"].append(metrics["赔率"])
                param_scores[(fast, slow)]["max_dds"].append(metrics["最大回撤 (%)"])
                param_scores[(fast, slow)]["sharpe"].append(metrics["稳健夏普比率"])
                param_scores[(fast, slow)]["r_dd"].append(metrics["R立方"])
                if baseline_return is not None:
                    param_scores[(fast, slow)]["return_vs_baseline"].append(
                        metrics["回归年度回报率 (%)"] - baseline_return)
                    param_scores[(fast, slow)]["dd_vs_baseline"].append(
                        baseline_dd - metrics["最大回撤 (%)"])
                    param_scores[(fast, slow)]["r_dd_vs_baseline"].append(
                        metrics["R立方"] - baseline_r_dd)

    # ---- 构建汇总表 ----
    summary = []
    for (fast, slow), scores in param_scores.items():
        summary.append({
            "ema_fast": fast,
            "ema_slow": slow,
            "avg_return": round(np.mean(scores["returns"]), 2) if scores["returns"] else 0,
            "avg_expect_rates": round(np.mean(scores["expect_rates"]), 1) if scores["expect_rates"] else 0,
            "avg_profit_factor": round(np.mean(scores["profit_factors"]), 2) if scores["profit_factors"] else 0,
            "avg_max_dd": round(np.mean(scores["max_dds"]), 2) if scores["max_dds"] else 0,
            "avg_sharpe": round(np.mean(scores["sharpe"]), 1) if scores["sharpe"] else 0,
            "avg_r_dd": round(np.mean(scores["r_dd"]), 1) if scores["r_dd"] else 0,
            "avg_return_improve": round(np.mean(scores["return_vs_baseline"]), 2) if scores["return_vs_baseline"] else 0,
            "avg_dd_improve": round(np.mean(scores["dd_vs_baseline"]), 2) if scores["dd_vs_baseline"] else 0,
            "avg_r_dd_improve": round(np.mean(scores["r_dd_vs_baseline"]), 2) if scores["r_dd_vs_baseline"] else 0,
        })

    summary_df = pd.DataFrame(summary)

    # ---- 综合评分 ----
    def _norm(s):
        mn, mx = s.min(), s.max()
        if mx == mn:
            return pd.Series(0.5, index=s.index)
        return (s - mn) / (mx - mn)

    summary_df["score_return"] = _norm(summary_df["avg_return"])
    summary_df["score_r_dd"] = _norm(summary_df["avg_r_dd"])
    summary_df["score_pf"] = _norm(summary_df["avg_profit_factor"])
    summary_df["score_er"] = _norm(summary_df["avg_expect_rates"])
    summary_df["score_dd"] = _norm(1 - summary_df["avg_max_dd"] / 100)
    summary_df["score_sharpe"] = _norm(summary_df["avg_sharpe"])
    summary_df["score_return_improve"] = _norm(summary_df["avg_return_improve"])

    # 最优参数评分权重：期望收益率20 % + 回归年度回报率25 % + R立方30 % + 稳健夏普比率25 % ？可调整
    summary_df["composite_score"] = (
        summary_df["score_r_dd"] * 0.30 +
        summary_df["score_return"] * 0.25 +
        summary_df["score_er"] * 0.20 +
        summary_df["score_sharpe"] * 0.25
    )

    summary_df = summary_df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    summary_df["rank"] = range(1, len(summary_df) + 1)
    return summary_df


# ============================================================
# 主流程
# ============================================================
def main():
    token = os.environ.get("TUSHARE_TOKEN")
    if not token:
        print("❌ 请先设置环境变量 TUSHARE_TOKEN")
        return
    pro = ts.pro_api(token)

    # ---- 1. 筛选流动性 >1亿 ----
    currency = "CNY" if MARKET == "A" else "HKD"
    market_label = "A股" if MARKET == "A" else "港股"
    print("=" * 80)
    print(f"🏆 近一年{market_label}日均成交额 >1亿")
    print("=" * 80)

    top10 = fetch_liquidity(pro, START_DATE, END_DATE)

    if MARKET == "A":
        sb = pro.stock_basic(list_status='L', fields='ts_code,name')
        name_map = dict(zip(sb['ts_code'], sb['name']))
    else:
        hk_basic = pro.hk_basic()
        name_map = {}
        for _, r in hk_basic.iterrows():
            name_map[r["ts_code"]] = r["name"]

    for i, (code, amt, _) in enumerate(top10, 1):
        name = name_map.get(code, "")
        print(f"  #{i} {code} {name}  日均成交额: {amt:>15,.0f} {currency}")

    # ---- 2. EMA 参数扫描 ----
    print(f"\n{'=' * 80}")
    print(f"📊 EMA 趋势过滤器参数扫描 ({market_label} 加仓做多)")
    print(f"   快线: {FAST_PERIODS}")
    print(f"   慢线: {SLOW_PERIODS}")
    print(f"   唐奇安通道: entry={ENTRY_PERIOD}, exit={EXIT_PERIOD}, atr={ATR_PERIOD}")
    print(f"   加仓: max_pyramid={MAX_PYRAMID}, skip_first_add={SKIP_FIRST_ADD}")
    print(f"{'=' * 80}\n")
    summary = run_parameter_scan(top10,name_map)

    # ---- 3. 输出排名 ----
    print(f"\n{'=' * 80}")
    print("🏆 综合排名 (评分权重: 回报改善35% + 盈亏比20% + 绝对回报15% + 回撤控制15% + 胜率10% + 交易5%)")
    print(f"{'=' * 80}")
    cols = ["rank", "ema_fast", "ema_slow", "avg_return", "avg_return_improve",
            "avg_win_rate", "avg_profit_factor", "avg_max_dd", "avg_trades",
            "avg_adds", "composite_score"]
    print(summary[cols].head(25).to_string(index=False))

    # ---- 4. 分组统计 ----
    print(f"\n{'=' * 80}")
    print("📊 按慢线分组")
    print(f"{'=' * 80}")
    for slow in [0] + SLOW_PERIODS:
        sub = summary[summary["ema_slow"] == slow]
        if sub.empty:
            continue
        top3_fast = sub.head(3)["ema_fast"].tolist()
        label = "无过滤器" if slow == 0 else f"EMA_Slow={slow}"
        print(f"  {label:<18} 组合:{len(sub):>2}  均分:{sub['composite_score'].mean():.4f}  "
              f"均回报改善:{sub['avg_return_improve'].mean():>+6.1f}%  Top快线:{top3_fast}")

    print(f"\n{'=' * 80}")
    print("📊 按快线分组")
    print(f"{'=' * 80}")
    for fast in [0] + FAST_PERIODS:
        sub = summary[summary["ema_fast"] == fast]
        if sub.empty:
            continue
        top3_slow = sub.head(3)["ema_slow"].tolist()
        label = "无过滤器" if fast == 0 else f"EMA_Fast={fast}"
        print(f"  {label:<18} 组合:{len(sub):>2}  均分:{sub['composite_score'].mean():.4f}  "
              f"均回报改善:{sub['avg_return_improve'].mean():>+6.1f}%  Top慢线:{top3_slow}")

    # ---- 5. 最优建议 ----
    best = summary[summary["ema_fast"] > 0].iloc[0]
    print(f"\n{'=' * 80}")
    print("🎯 最优参数建议")
    print(f"{'=' * 80}")
    print(f"  推荐 EMA_Fast = {best['ema_fast']}, EMA_Slow = {best['ema_slow']}")
    print(f"  综合评分: {best['composite_score']:.4f}")
    print(f"  平均回报: {best['avg_return']}% (相对基准改善 {best['avg_return_improve']:+.1f}%)")
    print(f"  平均胜率: {best['avg_win_rate']}%")
    print(f"  平均盈亏比: {best['avg_profit_factor']}")
    print(f"  平均回撤: {best['avg_max_dd']}%")
    print(f"  平均交易次数: {best['avg_trades']}")

    # ---- 6. 保存 ----
    summary.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ 结果已保存至 {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
