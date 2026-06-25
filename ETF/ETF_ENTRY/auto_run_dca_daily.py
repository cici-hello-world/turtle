#!/usr/bin/env python3
"""自动化 每日 DCA 定投三步流程：
  1. 针对不同周期（2/3/5年）滚动回测，按策略选取最有预测价值的窗口
  2. 每日定投无需选日，跳过此步
  3. 用最佳窗口运行完整每日定投回测

策略（切换顶部 STRATEGY 变量）：
  - conservative: 保守型，偏爱稳健可靠的窗口
  - balanced:     均衡型，回报与风险并重（默认）
  - aggressive:   激进型，偏爱高回报窗口

用法：
  python ETF_ENTRY/auto_run_dca_daily.py
"""

import numpy as np
import matplotlib.pyplot as plt

from ETF.ETF_ENTRY.rolling_dca_backtest import (
    fetch_full_data,
    pick_widows_daily,
)
from ETF.ETF_ENTRY.run_dca_backtest import run_result_daily

# ============================================================
# 配置
# ============================================================
YEARS_LIST = [2, 3, 5]
STRATEGY = "balanced"  # conservative | balanced | aggressive
TS_CODE = "513500.SH"
STOCK_NAME = "标普500ETF"


# ============================================================
# Main
# ============================================================
def main():
    strategy_names = {"conservative": "保守型", "balanced": "均衡型", "aggressive": "激进型"}
    print(f"🎯 策略模式: {strategy_names.get(STRATEGY, STRATEGY)}")

    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "PingFang SC"]
    plt.rcParams["axes.unicode_minus"] = False

    # ============================================================
    # 0. 获取全量数据
    # ============================================================
    df_full = fetch_full_data(TS_CODE, STOCK_NAME)

    data_start = df_full["Date"].iloc[0].strftime("%Y%m%d")
    data_end = df_full["Date"].iloc[-1].strftime("%Y%m%d")
    print(f"\n📅 数据范围: {data_start} → {data_end}")

    # ============================================================
    # 1. 循环周期list，pick_windows（每日定投版）
    # ============================================================
    period_results = []
    for wy in YEARS_LIST:
        print(f"\n{'=' * 65}")
        print(f"🔍 滚动回测（每日定投） —— {wy}年窗口")
        print(f"{'=' * 65}")

        df_windows_pick, pick_xirr, start, end = pick_widows_daily(
            df_full, wy, TS_CODE, STOCK_NAME, strategy=STRATEGY)

        if df_windows_pick is None:
            print(f"  ⚠️ {wy}年窗口数据不足，跳过")
            continue

        period_results.append({
            "years": wy,
            "df": df_windows_pick,
            "xirr": pick_xirr,
        })
        xirr_str = f"{pick_xirr:.2f}%" if not np.isnan(pick_xirr) else "N/A"
        print(f"  → {wy}年窗口选取 XIRR: {xirr_str}"
              f"（{start.strftime('%Y-%m-%d')} → {end.strftime('%Y-%m-%d')}）")

    if not period_results:
        print("❌ 没有足够数据生成任何窗口，退出")
        return

    # ============================================================
    # 2. 每日定投无需选日，直接用中位数 XIRR 对应的周期窗口
    # ============================================================
    sorted_by_xirr = sorted(period_results, key=lambda r: r["xirr"])
    best = sorted_by_xirr[len(sorted_by_xirr) // 2]

    print(f"\n{'=' * 65}")
    print(f"🏆 选定周期: {best['years']}年 (选取XIRR: {best['xirr']:.2f}%)")
    print(f"  每日定投无需选日，每个交易日均为定投日")
    print(f"{'=' * 65}")

    # ============================================================
    # 3. 用1返回的df_windows_pick去调run_result_daily
    # ============================================================
    print(f"\n{'=' * 65}")
    print(f"🚀 运行最终回测: {best['years']}年周期, 每日定投100元")
    print(f"{'=' * 65}")
    run_result_daily(best["df"], tc=TS_CODE, sn=STOCK_NAME, strategy=STRATEGY)


if __name__ == "__main__":
    main()
