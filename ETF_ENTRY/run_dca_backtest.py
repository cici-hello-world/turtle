#!/usr/bin/env python3
"""ETF 每月定投回测 —— 前 5 年每月 23 日定投，计算 XIRR 及全套统计指标
第三步生成投资预测
用法：
  python backtest_entry/run_dca_backtest.py

依赖：tushare, pandas, numpy, matplotlib
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from pandas import DataFrame

from backtest_entry.dca_utils import find_investment_dates, simulate_dca, _compute_xirr
from ETF_ENTRY.optimize_dca_day import START_DATE, END_DATE
from core.config import BacktestConfig
from core.data import fetch_data
from core.statistics import (
    _compute_drawdown_duration,
    _compute_regression_metrics,
    _compute_robust_sharpe,
    _compute_r_cubed,
)

# ============================================================
# 配置
# ============================================================
TS_CODE = "513500.SH"       # 标普500ETF
STOCK_NAME = "标普500ETF"
MONTHLY_AMOUNT = 3000.0     # 每月定投金额（元）
TARGET_DAY = 7             # 每月定投日
PER_SHARES = 100            # 每手股数
YEARS = 5                   # 回测年数

out_dir = os.path.dirname(os.path.dirname(__file__)) + "/ETF_dca_result"
os.makedirs(out_dir, exist_ok=True)
# ============================================================
# 1. 统计计算
# ============================================================
def calc_dca_stats(sim_result: dict, df: pd.DataFrame) -> dict:
    """基于 DCA 权益曲线计算全套统计指标"""
    equity_series = sim_result["equity_series"]
    equity_dates = sim_result["equity_dates"]
    total_invested = sim_result["total_invested"]
    final_value = sim_result["final_value"]
    first_date = equity_dates.iloc[0]
    last_date = equity_dates.iloc[-1]
    total_years = (last_date - first_date).days / 365.25

    # 总收益
    total_pnl = final_value - total_invested
    total_return = (total_pnl / total_invested) * 100 if total_invested > 0 else 0

    # XIRR：基于每笔投入现金流计算内部收益率
    investments_df = sim_result["investments"]
    cashflows = []
    cf_dates = []
    for _, inv in investments_df.iterrows():
        if inv["买入手数"] > 0:
            cashflows.append(-inv["买入金额"])
            cf_dates.append(inv["投资日期"])
    # 期末市值作为正现金流（假设最后一日清仓）
    cashflows.append(final_value)
    cf_dates.append(last_date)
    xirr = _compute_xirr(cashflows, cf_dates) if len(cashflows) > 1 else 0.0

    # 最大回撤
    cummax = equity_series.cummax()
    drawdown = (equity_series - cummax) / cummax
    max_dd = drawdown.min() * 100

    # 衰落持续期
    dd_duration = _compute_drawdown_duration(equity_series, equity_dates)

    # 夏普比率
    if len(equity_series) > 1:
        returns = equity_series.pct_change().dropna()
        if len(returns) > 0 and returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            sharpe = 0
    else:
        sharpe = 0

    # 稳健夏普
    robust_sharpe = _compute_robust_sharpe(equity_series)

    # 回归年度回报率 & R²
    reg_return, r_squared = _compute_regression_metrics(equity_series, equity_dates)

    # R立方
    r_cubed = _compute_r_cubed(equity_series, reg_return, equity_dates)

    # MAR比率
    mar_ratio = (xirr * 100) / abs(max_dd) if abs(max_dd) > 0.01 else 0

    # 每期定投盈亏（以最后一日清仓价计算每期盈亏）
    investments = sim_result["investments"]
    final_close = sim_result["last_close"]
    win_count, loss_count = 0, 0
    for _, inv in investments.iterrows():
        if inv["买入手数"] == 0:
            continue
        avg_cost = inv["买入金额"] / (inv["买入手数"] * PER_SHARES) if inv["买入手数"] > 0 else 0
        if final_close > avg_cost:
            win_count += 1
        elif final_close < avg_cost:
            loss_count += 1
    total_inv_periods = win_count + loss_count
    win_rate = (win_count / total_inv_periods * 100) if total_inv_periods > 0 else 0

    # 期望收益率（胜率 × 赔率风格）
    total_win_pnl = 0
    total_loss_pnl = 0
    for _, inv in investments.iterrows():
        if inv["买入手数"] == 0:
            continue
        shares_bought = inv["买入手数"] * PER_SHARES
        pnl = shares_bought * (final_close - inv["买入金额"] / shares_bought)
        if pnl > 0:
            total_win_pnl += pnl
        else:
            total_loss_pnl += abs(pnl)
    avg_win = total_win_pnl / win_count if win_count > 0 else 0
    avg_loss = total_loss_pnl / loss_count if loss_count > 0 else 0
    win_loss_ratio = (avg_win / avg_loss) if avg_loss > 0 else 0
    avg_trade = (avg_win + avg_loss) / 2 if (win_count + loss_count) > 0 else 0
    expectation = (win_rate / 100 * win_loss_ratio) - (100 - win_rate) / 100 if total_inv_periods > 0 else 0

    return {
        "总交易": total_inv_periods,
        "胜率 (%)": round(win_rate, 2),
        "赔率": round(win_loss_ratio, 4),
        "期望收益率": round(expectation, 4),
        "盈利次数": win_count,
        "亏损次数": loss_count,
        "总盈亏 (¥)": round(total_pnl, 2),
        "总收益率 (%)": round(total_return, 2),
        "最大回撤 (%)": round(max_dd, 2),
        "最终市值 (¥)": round(final_value, 2),
        "总投入 (¥)": round(total_invested, 2),
        "夏普比率": round(sharpe, 2),
        "平均盈亏 (¥)": round(avg_trade, 2),
        "XIRR (%)": round(xirr * 100, 2),
        "MAR比率": round(mar_ratio, 2),
        "衰落持续期 (天)": dd_duration,
        "回归年度回报率 (%)": round(reg_return * 100, 2),
        "R平方": round(r_squared, 4),
        "稳健夏普比率": round(robust_sharpe, 2),
        "R立方": round(r_cubed, 2),
        "first_date": first_date,
        "last_date": last_date,
        "total_years": round(total_years, 2),
    }


# ============================================================
# 4. 输出
# ============================================================
def print_dca_summary(stats: dict, investments: pd.DataFrame,tc,sn,td):
    """打印定投明细和统计摘要，并保存到 Excel（两个 sheet）"""
    # —— 保存到 Excel ——
    stats_df = pd.DataFrame([
        {"指标": k, "数值": v}
        for k, v in stats.items()
        if k not in ("first_date", "last_date")
    ])
    xlsx_path = os.path.join(out_dir, f"dca_{pd.to_datetime(stats["first_date"]).strftime("%Y%m")}_{pd.to_datetime(stats["last_date"]).strftime("%Y%m")}_{sn}.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        investments.to_excel(writer, sheet_name="每月定投明细", index=False)
        stats_df.to_excel(writer, sheet_name="统计摘要", index=False)
    print(f"\n📁 数据已保存至：{xlsx_path}")

    # —— 每月投资明细 ——
    print(f"\n{'=' * 100}")
    print("📋 每月定投明细")
    print(f"{'=' * 100}")
    print(investments.to_string(index=False))

    # —— 统计摘要 ——
    print(f"\n{'=' * 60}")
    print("📊 定投回测统计摘要")
    print(f"{'=' * 60}")
    print(f"  ETF: {sn} ({tc})")
    print(f"  回测区间: {stats['first_date'].strftime('%Y-%m-%d')} → {stats['last_date'].strftime('%Y-%m-%d')}")
    print(f"  回测年数: {stats['total_years']} 年")
    print(f"  定投策略: 每月 {td} 日定投 {MONTHLY_AMOUNT:,.0f} 元"
          f"（非交易日顺延）")
    print()

    rows = [
        ("💰 总投入", f"{stats['总投入 (¥)']:,.2f} 元"),
        ("📈 最终市值", f"{stats['最终市值 (¥)']:,.2f} 元"),
        ("💵 总盈亏", f"{stats['总盈亏 (¥)']:+,.2f} 元"),
        ("📊 总收益率", f"{stats['总收益率 (%)']:+.2f}%"),
        ("📐 XIRR（年化）", f"{stats['XIRR (%)']:+.2f}%"),
        ("📉 最大回撤", f"{stats['最大回撤 (%)']:.2f}%"),
        ("⏱️ 衰落持续期", f"{stats['衰落持续期 (天)']} 天"),
        ("🎯 胜率（盈利期数/总期数）", f"{stats['胜率 (%)']:.2f}%"),
        ("🎲 赔率", f"{stats['赔率']:.4f}"),
        ("📐 夏普比率", f"{stats['夏普比率']:.2f}"),
        ("📐 稳健夏普比率", f"{stats['稳健夏普比率']:.2f}"),
        ("📈 回归年度回报率", f"{stats['回归年度回报率 (%)']:+.2f}%"),
        ("📈 R平方", f"{stats['R平方']:.4f}"),
        ("📈 MAR比率", f"{stats['MAR比率']:.2f}"),
        ("🧊 R立方", f"{stats['R立方']:.2f}"),
        ("📋 定投期数", f"{stats['总交易']} 期"),
        ("✅ 盈利期数", f"{stats['盈利次数']} 期"),
        ("❌ 亏损期数", f"{stats['亏损次数']} 期"),
    ]
    for label, value in rows:
        print(f"  {label:<20}  {value}")


def plot_dca(df: pd.DataFrame, sim_result: dict, stats: dict, tc, sn, td):
    """绘制定投权益曲线"""
    equity_series = sim_result["equity_series"]
    invest_dates = pd.to_datetime(sim_result["investments"]["投资日期"])

    fig, ax1 = plt.subplots(figsize=(18, 7))

    # 权益曲线
    ax1.plot(equity_series.index, equity_series.values,
             color="#2e86c1", linewidth=1.5, label="权益曲线（市值 + 现金）")
    ax1.fill_between(equity_series.index, equity_series.values,
                     sim_result["total_invested"], alpha=0.15, color="#2e86c1")

    # 累计投入（阶梯线）
    invested_curve = []
    cum = 0
    invest_df = sim_result["investments"]
    for _, row in df.sort_values("Date").iterrows():
        d = row["Date"]
        cum = invest_df[invest_df["投资日期"] <= d.strftime("%Y-%m-%d")]["买入金额"].sum()
        invested_curve.append(cum)
    ax1.plot(df.sort_values("Date")["Date"], invested_curve,
             color="#e67e22", linewidth=1.2, linestyle="--", alpha=0.8, label="累计投入")

    # 定投点标记
    for _, inv in sim_result["investments"].iterrows():
        if inv["买入手数"] > 0:
            ax1.axvline(pd.to_datetime(inv["投资日期"]), color="#27ae60",
                       linewidth=0.5, alpha=0.3)

    # 标注最终值和 XIRR
    final_val = stats["最终市值 (¥)"]
    final_date = equity_series.index[-1]
    ax1.annotate(
        f"最终: ¥{final_val:,.0f}\nXIRR: {stats['XIRR (%)']:+.2f}%",
        xy=(final_date, equity_series.values[-1]),
        xytext=(15, 30), textcoords="offset points",
        fontsize=10, fontweight="bold", color="#2e86c1",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85),
        arrowprops=dict(arrowstyle="->", color="#2e86c1", lw=1.2),
    )

    ax1.set_ylabel("市值（元）", fontsize=11)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y/%m"))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=9)

    ax1.set_title(
        f"{sn} ({tc}) 每月{td}日定投 {MONTHLY_AMOUNT:,.0f}元 "
        f"—— {stats['total_years']}年回测 | XIRR: {stats['XIRR (%)']:+.2f}%",
        fontsize=14, fontweight="bold", pad=18,
    )
    ax1.legend(loc="upper left", fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    out_path = os.path.join(out_dir,
                            f"dca_{pd.to_datetime(stats["first_date"]).strftime("%Y%m")}_{pd.to_datetime(stats["last_date"]).strftime("%Y%m")}_{sn}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n📈 图表已保存至：{out_path}")
    plt.show()

def run_result(df: DataFrame,td=TARGET_DAY,tc:str=TS_CODE,sn:str=STOCK_NAME):
    # 2. 定位每月投资日
    invest_dates = find_investment_dates(df, td)
    if invest_dates.empty:
        print("❌ 未找到任何投资日，请检查数据")
        return
    print(f"✅ 共定位 {len(invest_dates)} 个定投日"
          f"（{invest_dates['Date'].iloc[0].strftime('%Y-%m-%d')} → "
          f"{invest_dates['Date'].iloc[-1].strftime('%Y-%m-%d')}）")

    # 3. DCA 模拟
    sim_result = simulate_dca(df, invest_dates, MONTHLY_AMOUNT, PER_SHARES)

    # 4. 统计
    stats = calc_dca_stats(sim_result, df)

    # 5. 输出
    print_dca_summary(stats, sim_result["investments"], tc, sn, td)
    plot_dca(df, sim_result, stats, tc, sn, td)


# ============================================================
# Main
# ============================================================
def main():
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "PingFang SC"]
    plt.rcParams["axes.unicode_minus"] = False

    config = BacktestConfig(
        ts_code=TS_CODE,
        stock_name=STOCK_NAME,
        market="ETF",
        years=YEARS,
    )

    # 1. 获取数据
    print(f"⏳ 正在获取 {STOCK_NAME} ({TS_CODE}) 近 {YEARS} 年数据...")
    df = fetch_data(
        ts_code=config.ts_code,
        stock_name=config.stock_name,
        market=config.market,
        start_date=START_DATE,
        end_date=END_DATE,
        exchange=config.exchange,
    )

    run_result(df)



if __name__ == "__main__":
    main()
