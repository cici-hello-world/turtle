#!/usr/bin/env python3
"""扫描 1-31 日定投日，找出 XIRR 最大的那一天
第二步，查询定投日
用法：
  python backtest_entry/optimize_dca_day.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pandas import DataFrame

from core.config import BacktestConfig
from core.data import fetch_data
from ETF.dca_utils import find_investment_dates, simulate_dca, _compute_xirr

# ============================================================
# 配置
# ============================================================
TS_CODE = "513500.SH"
STOCK_NAME = "标普500ETF"
MONTHLY_AMOUNT = 3000.0
PER_SHARES = 100
# ===================确定周期后输入参数===========================
YEARS = 5
START_DATE="20191215"
END_DATE="20241215"

out_dir = os.path.dirname(os.path.dirname(__file__)) + "/ETF_dca_day_pick"
os.makedirs(out_dir, exist_ok=True)


def pick_day(df: DataFrame, wy: int = YEARS, tc: str = TS_CODE, sn: str = STOCK_NAME, strategy: str = "balanced"):
    # 2. 扫描 1-31 日
    print(f"\n🔍 扫描 1-31 日定投日 ...")
    print(f"{'=' * 85}")
    print(
        f"{'定投日':<6} {'投资期数':<8} {'总投入':>12} {'最终市值':>12} {'总收益率':>10} {'XIRR':>10} {'最大回撤':>10}")
    print(f"{'-' * 85}")

    results = []
    for day in range(1, 32):
        invest_dates = find_investment_dates(df, day)
        if invest_dates.empty:
            continue

        sim = simulate_dca(df, invest_dates, MONTHLY_AMOUNT, PER_SHARES)

        first_date = sim["equity_dates"].iloc[0]
        last_date = sim["equity_dates"].iloc[-1]
        total_return = (sim["final_value"] - sim["total_invested"]) / sim["total_invested"] * 100

        # 最大回撤
        eq = sim["equity_series"]
        cummax = eq.cummax()
        max_dd = ((eq - cummax) / cummax).min() * 100

        # XIRR
        inv_df = sim["investments"]
        cashflows = []
        cf_dates = []
        for _, inv in inv_df.iterrows():
            if inv["买入手数"] > 0:
                cashflows.append(-inv["买入金额"])
                cf_dates.append(inv["投资日期"])
        cashflows.append(sim["final_value"])
        cf_dates.append(last_date)
        xirr = _compute_xirr(cashflows, cf_dates) if len(cashflows) > 1 else 0.0

        results.append({
            "定投日": day,
            "投资期数": len(invest_dates),
            "总投入": sim["total_invested"],
            "最终市值": sim["final_value"],
            "总收益率": total_return,
            "XIRR": xirr * 100,
            "最大回撤": max_dd,
            "sim": sim,
        })

        marker = ""
        if len(results) >= 2:
            if xirr > max(r["XIRR"] / 100 for r in results[:-1]):
                marker = " ⬅️ 新高"

        print(f"每月{day:>2}日  {len(invest_dates):>6}期  "
              f"{sim['total_invested']:>10,.0f}  {sim['final_value']:>10,.0f}  "
              f"{total_return:>8.2f}%  {xirr * 100:>8.2f}%  {max_dd:>8.2f}%{marker}")

    # 3. 排名
    results.sort(key=lambda x: x["XIRR"], reverse=True)

    print(f"\n{'=' * 85}")
    print("🏆 定投日 XIRR 排名 (Top 10)")
    print(f"{'=' * 85}")
    print(f"{'排名':<4} {'定投日':<6} {'投资期数':<8} {'总收益率':>10} {'XIRR':>10} {'最大回撤':>10}")
    print(f"{'-' * 60}")

    for rank, r in enumerate(results[:10], 1):
        print(f"#{rank:<3} 每月{r['定投日']:>2}日  {r['投资期数']:>6}期  "
              f"{r['总收益率']:>8.2f}%  {r['XIRR']:>8.2f}%  {r['最大回撤']:>8.2f}%")

    # 4. 可视化 —— 全部 31 天的 XIRR 柱状图
    best = max([r for r in results if r["投资期数"] >= wy * 12], key=lambda x: x["XIRR"])
    worst = results[-1]
    print(f"\n📌 最优: 每月 {best['定投日']} 日 → XIRR {best['XIRR']:.2f}%")
    print(f"📌 最差: 每月 {worst['定投日']} 日 → XIRR {worst['XIRR']:.2f}%")
    print(f"📌 差异: {best['XIRR'] - worst['XIRR']:.2f} 个百分点")

    # —— 画图 ——
    days = [r["定投日"] for r in results]
    xirrs = [r["XIRR"] for r in results]
    colors = ["#e74c3c" if c == max(xirrs) else
              "#95a5a6" if c == min(xirrs) else "#3498db" for c in xirrs]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

    # 上图：XIRR 柱状图
    bars = ax1.bar(days, xirrs, color=colors, edgecolor="white", linewidth=0.5)
    ax1.axhline(y=np.mean(xirrs), color="#e67e22", linestyle="--", linewidth=1,
                label=f"均值 XIRR: {np.mean(xirrs):.2f}%")
    ax1.set_xlabel("每月定投日", fontsize=11)
    ax1.set_ylabel("XIRR (%)", fontsize=11)
    ax1.set_title(f"{sn} ({tc}) 1-31日定投 XIRR 对比", fontsize=14, fontweight="bold")
    ax1.set_xticks(range(1, 32))
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle="--", axis="y")

    # 标注最优点
    ax1.annotate(f"最优: {best['定投日']}日\nXIRR: {best['XIRR']:.2f}%",
                 xy=(best["定投日"], best["XIRR"]),
                 xytext=(0, 20), textcoords="offset points",
                 fontsize=9, fontweight="bold", color="#e74c3c",
                 ha="center",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))

    # 下图：最优 vs 最差 权益曲线对比
    best_sim = best["sim"]
    worst_sim = worst["sim"]

    ax2.plot(best_sim["equity_series"].index, best_sim["equity_series"].values,
             color="#e74c3c", linewidth=1.5, label=f"每月{best['定投日']}日 (XIRR {best['XIRR']:.2f}%)")
    ax2.plot(worst_sim["equity_series"].index, worst_sim["equity_series"].values,
             color="#95a5a6", linewidth=1.5, label=f"每月{worst['定投日']}日 (XIRR {worst['XIRR']:.2f}%)")
    # 中位数曲线
    median_idx = len(results) // 2
    median_sim = results[median_idx]["sim"]
    median_day = results[median_idx]["定投日"]
    ax2.plot(median_sim["equity_series"].index, median_sim["equity_series"].values,
             color="#3498db", linewidth=1.0, alpha=0.6,
             label=f"每月{median_day}日 (中位数 XIRR {results[median_idx]['XIRR']:.2f}%)")

    ax2.set_title("最优 / 中位数 / 最差 定投日权益曲线对比", fontsize=13, fontweight="bold")
    ax2.set_ylabel("市值（元）", fontsize=11)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"_{wy}年周期_{sn}_{strategy}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n📈 图表已保存至：{out_path}")

    # 保存 Excel
    xlsx_path = os.path.join(out_dir, f"_{wy}年周期_{sn}_{strategy}.xlsx")
    ranking_df = pd.DataFrame(results).drop(columns=["sim"])
    best_inv_df = best["sim"]["investments"]
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        ranking_df.to_excel(writer, sheet_name="定投日排名", index=False)
        best_inv_df.to_excel(writer, sheet_name="最优定投日明细", index=False)
    print(f"📄 Excel 已保存至：{xlsx_path}")
    plt.show()
    day_pick = best["定投日"]
    return day_pick

# ============================================================
# Main
# ============================================================
def main():
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "PingFang SC"]
    plt.rcParams["axes.unicode_minus"] = False

    config = BacktestConfig(
        ts_code=TS_CODE, stock_name=STOCK_NAME,
        market="ETF", years=YEARS,
    )

    # 1. 获取数据（只拉一次）
    print(f"⏳ 正在获取 {STOCK_NAME} ({TS_CODE}) 近 {YEARS} 年数据...")
    df = fetch_data(
        ts_code=config.ts_code, stock_name=config.stock_name,
        market=config.market, start_date=START_DATE,
        end_date=END_DATE, exchange=config.exchange,
    )

    pick_day(df)




if __name__ == "__main__":
    main()
