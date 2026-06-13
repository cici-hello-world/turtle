#!/usr/bin/env python3
"""自动化 DCA 定投三步流程：
  1. 针对不同周期（2/3/5/8/10年）滚动回测，找最佳窗口
  2. 用中位数 XIRR 对应的周期窗口扫描最佳定投日
  3. 用最佳定投日运行完整回测

用法：
  python ETF_ENTRY/auto_run_dca.py
"""

import numpy as np
import matplotlib.pyplot as plt

from ETF_ENTRY.rolling_dca_backtest import (
    fetch_full_data,
    pick_widows,
)
from ETF_ENTRY.optimize_dca_day import pick_day
from ETF_ENTRY.run_dca_backtest import run_result

# ============================================================
# 配置
# ============================================================
YEARS_LIST = [2, 3, 5]
STEP_MONTHS = 1
# TS_CODE = "513300.SH"
# STOCK_NAME = "纳斯达克ETF"
TS_CODE = "513500.SH"
STOCK_NAME = "标普500ETF"

# ============================================================
# Main
# ============================================================
def main():
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
    # 1. 循环周期list，pick_windows
    #    对每个窗口年限跑滚动回测，收集 (years, df_windows_pick, xirr)
    # ============================================================
    period_results = []  # list[dict]: years, df, xirr
    for wy in YEARS_LIST:
        print(f"\n{'=' * 65}")
        print(f"🔍 滚动回测 —— {wy}年窗口")
        print(f"{'=' * 65}")

        df_windows_pick,mid_xirr,start,end = pick_widows(df_full,wy,TS_CODE,STOCK_NAME)
        period_results.append({
            "years": wy,
            "df": df_windows_pick,
            "xirr": mid_xirr,
        })
        xirr_str = f"{mid_xirr:.2f}%" if not np.isnan(mid_xirr) else "N/A"
        print(f"  → {wy}年窗口中位数 XIRR: {xirr_str}"
              f"（{start.strftime('%Y-%m-%d')} → {end.strftime('%Y-%m-%d')}）")

    # ============================================================
    # 2. 用在1返回的XIRR数组的中位数对应的周期和df_windows_pick去调pick_day
    # ============================================================
    # 按 XIRR 排序，取中位数位置对应的那组结果
    sorted_by_xirr = sorted(period_results, key=lambda r: r["xirr"])
    best = sorted_by_xirr[len(sorted_by_xirr) // 2]

    print(f"\n{'=' * 65}")
    print(f"🏆 选定中位数 XIRR 周期: {best['years']}年 "
          f"(XIRR: {best['xirr']:.2f}%)")
    print(f"{'=' * 65}")

    day_pick = pick_day(best["df"], wy=best["years"], tc=TS_CODE, sn=STOCK_NAME)
    print(f"\n📌 最佳定投日: 每月 {day_pick} 日")

    # ============================================================
    # 3. 用2返回的day_pick和1返回的df_windows_pick去调run_result
    # ============================================================
    print(f"\n{'=' * 65}")
    print(f"🚀 运行最终回测: {best['years']}年周期, 每月{day_pick}日定投")
    print(f"{'=' * 65}")
    run_result(best["df"], td=day_pick,tc=TS_CODE, sn=STOCK_NAME)


if __name__ == "__main__":
    main()
