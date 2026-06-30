#!/usr/bin/env python3
"""回测分析报告生成器

从网格搜索 + 8种模拟行情回测结果中提取数据，生成综合分析报告。

用法:
  python backtest_report.py                # 默认: 02015.HK
  python backtest_report.py 03690.HK       # 分析其他标的
"""

import sys
import os
import pandas as pd
import numpy as np

RESULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "result")


# ═══════════════════════════════════════════════════════════════
# 场景概率 (来自 scenario_analysis.py 的宏观估算)
# ═══════════════════════════════════════════════════════════════
SCENARIO_PROBS = {
    "缓慢阴跌": 0.35,
    "震荡向下": 0.20,
    "横盘震荡": 0.15,
    "V形反转":  0.10,
    "震荡向上": 0.10,
    "单边下跌": 0.05,
    "倒V反转":  0.03,
    "单边上涨": 0.02,
}


def generate_report(ts_code: str = "02015.HK"):
    """生成回测分析报告并打印。

    优先读取网格搜索结果(pm1_grid_search_results.xlsx)，
    若无可读取独立回测报告(Backtest_Report_PM_*.xlsx)。
    """

    # ── 1. 真实数据回测结果 ──
    sim_dir = os.path.join(RESULT_DIR, "position_management_result")
    gs_path = os.path.join(RESULT_DIR, "pm1_grid_search_results.xlsx")

    # 尝试网格搜索
    real_from_grid = False
    best_params_available = False
    if os.path.exists(gs_path):
        gs = pd.read_excel(gs_path)
        # 检查是否有该ts_code的专属列或是否只有一只股票
        if "ts_code" not in gs.columns:
            # 只有一只股票的网格搜索 → 直接取最优
            best = gs.iloc[0]
            buy_hold = best["return"] - best.get("excess", 0)
            real_from_grid = True
            best_params_available = True

    if not real_from_grid:
        # 读取独立回测报告
        backtest_files = [f for f in os.listdir(sim_dir)
                          if f.startswith(f"Backtest_Report_PM_") and ts_code in f]
        if backtest_files:
            bt_path = os.path.join(sim_dir, backtest_files[0])
            bt_summary = pd.read_excel(bt_path, sheet_name="Summary")
            bt = dict(zip(bt_summary["指标"], bt_summary["数值"]))
            best = {
                "return": float(bt.get("总收益率 (%)", 0)),
                "excess": float(bt.get("超额收益 (%)", 0)),
                "max_dd": float(bt.get("最大回撤 (%)", 0)),
                "trades": int(bt.get("交易次数", 0)),
                "sharpe": float(bt.get("夏普比率", 0)),
                "robust_sharpe": float(bt.get("稳健夏普比率", 0)),
                "cagr": float(bt.get("CAGR (%)", 0)),
                "final_capital": float(bt.get("最终资金 (¥)", 0)),
                # 参数未知
                "pm_grid_down": "-", "pm_grid_up": "-", "pm_reduce_pct": "-",
                "pm_add_pct": "-", "pm_adx_threshold": "-",
                "pm_agg_k": "-",
            }
            buy_hold = best["return"] - best["excess"]
        else:
            print(f"  ⚠️ 未找到 {ts_code} 的真实回测数据")
            return

    # ── 2. 8种模拟行情回测结果 ──
    sim_dir = os.path.join(RESULT_DIR, "position_management_result")
    scenarios = list(SCENARIO_PROBS.keys())
    sim_rows = []
    for label in scenarios:
        path = os.path.join(sim_dir, f"simulate_{ts_code}_{label}.xlsx")
        try:
            summary = pd.read_excel(path, sheet_name="Summary")
            s = dict(zip(summary["指标"], summary["数值"]))
            sim_rows.append({
                "label": label,
                "ret": float(s.get("总收益率 (%)", 0)),
                "excess": float(s.get("超额收益 (%)", 0)),
                "pnl": float(s.get("总盈亏 (¥)", 0)),
                "dd": float(s.get("最大回撤 (%)", 0)),
                "trades": int(s.get("交易次数", 0)),
                "sharpe": float(s.get("夏普比率", 0)),
                "cagr": float(s.get("CAGR (%)", 0)),
                "final_cap": float(s.get("最终资金 (¥)", 0)),
                "win_rate": float(s.get("胜率 (%)", 0)),
            })
        except Exception as e:
            print(f"  ⚠️ {label}: {e}")

    sim_df = pd.DataFrame(sim_rows)

    # ══════════════════════════════════════════════════
    # 打印报告
    # ══════════════════════════════════════════════════
    print("=" * 95)
    print(f"  {ts_code} 仓位管理策略1 回测分析报告")
    print(f"  真实数据: 2024-06-28 → 2026-06-26  |  模拟: 未来126天")
    print(f"  初始资金: ¥40,000  每手: 100股")
    print("=" * 95)

    # ── 一、真实数据回测 ──
    print(f"\n{'─' * 65}")
    print("一、真实数据回测 (网格搜索最优)")
    print(f"{'─' * 65}")
    print(f"  买入持有收益:  {buy_hold:+.2f}%")
    print(f"  策略收益:      {best['return']:+.2f}%")
    print(f"  超额收益:      {best['excess']:+.2f}%")
    print(f"  最大回撤:      {best['max_dd']:.1f}%")
    print(f"  夏普比率:      {best['sharpe']:.2f}")
    print(f"  稳健夏普:      {best['robust_sharpe']:.2f}")
    print(f"  CAGR:          {best['cagr']:.1f}%")
    print(f"  最终资金:      ¥{best['final_capital']:,.0f}")
    print(f"  总交易:        {best['trades']:.0f}笔")
    if best_params_available:
        print(f"\n  最优参数:")
        gd = best.get('pm_grid_down', 0)
        gu = best.get('pm_grid_up', 0)
        rp = best.get('pm_reduce_pct', 0)
        ap = best.get('pm_add_pct', 0)
        k = best.get('pm_agg_k', 0)
        adx = best.get('pm_adx_threshold', 0)
        print(f"    趋势: gd={gd}  gu={gu}  rp={rp}  ap={ap}")
        print(f"    激进倍数: k={k}")
        print(f"    ADX阈值: {adx}")

    # ── 二、模拟行情回测 ──
    print(f"\n{'─' * 95}")
    print("二、8种模拟行情回测 (未来126天)")
    print(f"{'─' * 95}")
    print(f"{'场景':<10} {'收益%':>8} {'超额%':>8} {'盈亏¥':>10} {'回撤%':>7} "
          f"{'交易':>4} {'夏普':>6} {'CAGR%':>7} {'最终¥':>9}")
    print("-" * 85)
    for _, r in sim_df.iterrows():
        print(f"{r['label']:<10} {r['ret']:>+7.2f}% {r['excess']:>+7.2f}% "
              f"¥{r['pnl']:>+9,.0f} {r['dd']:>+6.1f}% {r['trades']:>4} "
              f"{r['sharpe']:>5.2f} {r['cagr']:>+6.1f}% ¥{r['final_cap']:>8,.0f}")

    # ── 三、概率加权预期 ──
    print(f"\n{'─' * 65}")
    print("三、概率加权预期 (宏观情景概率 × 各场景回测)")
    print(f"{'─' * 65}")

    w_ret = w_excess = w_dd = w_sharpe = 0.0
    for _, r in sim_df.iterrows():
        p = SCENARIO_PROBS.get(r["label"], 0)
        w_ret += r["ret"] * p
        w_excess += r["excess"] * p
        w_dd += r["dd"] * p
        w_sharpe += r["sharpe"] * p

    print(f"  概率加权预期收益:  {w_ret:+.1f}%")
    print(f"  概率加权超额收益:  {w_excess:+.1f}%")
    print(f"  概率加权最大回撤:  {w_dd:.1f}%")
    print(f"  概率加权夏普:      {w_sharpe:.2f}")

    best_scene = sim_df.loc[sim_df["ret"].idxmax()]
    worst_scene = sim_df.loc[sim_df["ret"].idxmin()]
    print(f"\n  最佳场景: {best_scene['label']} ({best_scene['ret']:+.1f}%, "
          f"概率{SCENARIO_PROBS[best_scene['label']]:.0%})")
    print(f"  最差场景: {worst_scene['label']} ({worst_scene['ret']:+.1f}%, "
          f"概率{SCENARIO_PROBS[worst_scene['label']]:.0%})")

    # ── 四、综合评分 ──
    print(f"\n{'─' * 75}")
    print("四、综合评分 (收益 × 概率)")
    print(f"{'─' * 75}")
    sorted_sim = sim_df.sort_values("ret", ascending=False)
    for _, r in sorted_sim.iterrows():
        p = SCENARIO_PROBS.get(r["label"], 0)
        weighted = r["ret"] * p
        bar = "█" * int(abs(weighted) * 10) if abs(weighted) < 5 else "█" * 50
        print(f"  {r['label']:<10} {r['ret']:>+7.2f}% × {p:.0%} = {weighted:>+7.2f}%  {bar}")

    # ── 五、结论 ──
    print(f"\n{'─' * 65}")
    print("五、结论")
    print(f"{'─' * 65}")
    print(f"  1. 真实回测: 策略比买入持有少亏 {abs(best['excess']):.1f}%")
    print(f"  2. 概率加权: 未来6月预期超额 {w_excess:+.1f}%，策略有望继续跑赢持有")
    print(f"  3. 最大风险: {worst_scene['label']}场景({SCENARIO_PROBS[worst_scene['label']]:.0%}概率) "
          f"可能再亏{abs(worst_scene['ret']):.1f}%")
    if best_params_available:
        print(f"  4. 关键参数: 趋势gd={gd} gu={gu} rp={rp}, 激进倍数k={k}, ADX阈值={adx}")
    else:
        print(f"  4. 参数: 非网格搜索最优，详见回测报告")
    print(f"  5. 风险提示: 模拟基于历史形态假设, 实际走势受基本面/情绪/政策多重影响")
    print(f"\n{'=' * 95}")


if __name__ == "__main__":
    ts_code = sys.argv[1] if len(sys.argv) > 1 else "02015.HK"
    generate_report(ts_code)
