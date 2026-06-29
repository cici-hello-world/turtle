#!/usr/bin/env python3
"""PM1 自适应网格策略 —— 全参数网格搜索

搜索所有可微调参数的最优组合，基于真实回测总收益率评分。

用法：
  python grid_search_pm.py
"""

import itertools
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.config import BacktestConfig
from core.data import fetch_data
from core.indicators import compute_indicators
from core.engine import run_backtest
from core.statistics import calc_stats


def grid_search():
    # ==================== 固定参数 ====================
    # base_kwargs = dict(
    #     ts_code="06181.HK",
    #     stock_name="老铺",
    #     market="HK",
    #     mode="position_management_1",
    #     per_shares=100,
    #     pm_start_date="20251001",
    #     start_date="20240628",
    #     user_dmi=True,
    #     pm_adaptive=True,
    #     pm_adaptive_grid=True,
    #     pm_dynamic_position=False,
    #     ema_fast=0,
    #     ema_slow=0,
    #     pm_initial_shares=3,
    #     initial_capital=40000,
    #     data_source="akshare",
    # )
    # base_kwargs = dict(
    #     ts_code="03690.HK",
    #     stock_name="美团",
    #     market="HK",
    #     mode="position_management_1",
    #     per_shares=100,
    #     pm_start_date="20220321",
    #     start_date="20180920",
    #     end_date="20220922",
    #     user_dmi=True,
    #     pm_adaptive=True,
    #     pm_adaptive_grid=True,
    #     pm_dynamic_position=False,
    #     ema_fast=0,
    #     ema_slow=0,
    #     pm_initial_shares=2,
    #     initial_capital=40000,
    #     data_source="akshare",
    # )
    # base_kwargs = dict(
    #     ts_code="02015.HK",
    #     stock_name="理想",
    #     market="HK",
    #     mode="position_management_1",
    #     per_shares=100,
    #     pm_start_date="20251001",
    #     start_date="20220813",
    #     user_dmi=True,
    #     pm_adaptive=True,
    #     pm_adaptive_grid=True,
    #     pm_dynamic_position=True,
    #     ema_fast=0,
    #     ema_slow=0,
    #     pm_initial_shares=25,
    #     initial_capital=40000,
    #     data_source="akshare",
    # )
    base_kwargs = dict(
            ts_code="09992.HK",
            stock_name="泡泡玛特",
            market="HK",
            mode="position_management_1",
            per_shares=200,
            pm_start_date="20231101",
            start_date="20201211",
            end_date="20240228",
            user_dmi=True,
            pm_adaptive=True,
            pm_adaptive_grid=True,
            pm_dynamic_position=True,
            ema_fast=0,
            ema_slow=0,
            pm_initial_shares=2,
            initial_capital=40000,
            data_source="akshare",
        )

    # ==================== 搜索空间 ====================
    pm_grid_down_vals       = [0.03, 0.05, 0.07, 0.10, 0.12]         # 5
    pm_grid_up_vals         = [0.03, 0.04, 0.05, 0.07, 0.10]         # 5
    pm_reduce_pct_vals      = [0.25, 0.35, 0.50, 0.75]               # 4
    pm_add_pct_vals         = [0.015, 0.02, 0.025, 0.03]             # 4
    pm_adx_threshold_vals   = [25, 30, 35, 40]                       # 4
    pm_grid_down_agg_vals   = [0.02, 0.03, 0.05]                     # 3
    pm_grid_up_agg_vals     = [0.02, 0.03, 0.04, 0.05]               # 4
    pm_reduce_pct_agg_vals  = [0.25, 0.35, 0.50, 0.75]               # 4
    pm_take_profit_vals     = [1.30]                                  # 1 (固定)
    # 全量: 5×5×3×4×4×2×2×3×1 = 14,400，约束过滤后约 8,000

    search_lists = [
        pm_grid_down_vals, pm_grid_up_vals, pm_reduce_pct_vals, pm_add_pct_vals,
        pm_adx_threshold_vals, pm_grid_down_agg_vals, pm_grid_up_agg_vals,
        pm_reduce_pct_agg_vals, pm_take_profit_vals,
    ]

    total = 1
    for lst in search_lists:
        total *= len(lst)
    print(f"🔍 搜索空间: {total:,} 种组合  (mod 由 agg 和趋势的中点自动计算)")
    print(f"   grid_down: {pm_grid_down_vals}")
    print(f"   grid_up:   {pm_grid_up_vals}")
    print(f"   reduce:    {pm_reduce_pct_vals}")
    print(f"   add:       {pm_add_pct_vals}")
    print(f"   adx_thr:   {pm_adx_threshold_vals}")
    print(f"   agg_dn:    {pm_grid_down_agg_vals}")
    print(f"   agg_up:    {pm_grid_up_agg_vals}")
    print(f"   agg_rp:    {pm_reduce_pct_agg_vals}")
    print(f"   take_prof: {pm_take_profit_vals}")

    # ==================== 预加载数据 ====================
    print("\n📥 获取数据 & 计算指标 ...")
    config0 = BacktestConfig(**base_kwargs)
    df_base = fetch_data(
        ts_code=config0.ts_code, stock_name=config0.stock_name, market=config0.market,
        start_date=config0.start_date, end_date=config0.end_date,
        exchange=config0.exchange, data_source=config0.data_source,
    )
    df_base = compute_indicators(
        df_base, entry_period=config0.entry_period, exit_period=config0.exit_period,
        atr_period=config0.atr_period, ema_fast=config0.ema_fast, ema_slow=config0.ema_slow,
        channel_type=config0.channel_type, bb_period=config0.bb_period, bb_k=config0.bb_k,
        am_ma_period=config0.am_ma_period, mode=config0.mode,
    )
    print(f"   数据行数: {len(df_base)}")

    df_pm = df_base[df_base["Date"] >= config0.pm_start_date]
    buy_hold_return = ((df_pm["Close"].iloc[-1] - df_pm["Close"].iloc[0])
                       / df_pm["Close"].iloc[0] * 100) if len(df_pm) >= 2 else 0.0
    print(f"   买入持有收益率 (自{config0.pm_start_date}): {buy_hold_return:+.2f}%\n")

    # ==================== 网格搜索 ====================
    best_return = -float("inf")
    best_result = None
    best_stats = None
    best_params = None
    results = []
    count = 0
    skipped = 0
    t0 = time.time()

    for gd, gu, rp, ap, adx, gda, gua, rpa, tp in itertools.product(*search_lists):
        count += 1
        # 约束: agg ≤ 趋势 (震荡市应比趋势市更激进)
        if gda > gd or gua > gu or rpa < rp:
            skipped += 1
            continue
        params = dict(
            pm_grid_down=gd, pm_grid_up=gu, pm_reduce_pct=rp, pm_add_pct=ap,
            pm_adx_threshold=adx, pm_grid_down_agg=gda, pm_grid_up_agg=gua,
            pm_reduce_pct_agg=rpa, pm_take_profit_ratio=tp,
        )
        config = BacktestConfig(**{**base_kwargs, **params})

        try:
            trades, equity, eq_dates = run_backtest(df_base.copy(), config)
            stats = calc_stats(trades, config.initial_capital, equity, eq_dates)
            total_return = stats["总收益率 (%)"]
            excess = total_return - buy_hold_return

            results.append({
                **params,
                "return": total_return,
                "excess": excess,
                "trades": stats["总交易"],
                "max_dd": stats["最大回撤 (%)"],
                "sharpe": stats["夏普比率"],
                "robust_sharpe": stats["稳健夏普比率"],
                "cagr": stats["CAGR (%)"],
                "final_capital": stats["最终资金 (¥)"],
            })

            marker = ""
            if total_return > best_return:
                best_return = total_return
                best_result = (gd, gu, rp, ap, adx, gda, gua, rpa, tp)
                best_stats = stats
                best_params = params
                marker = " ⭐"

            elapsed = time.time() - t0
            if count % 200 == 0 or count == 1:
                eta = (elapsed / count) * (total - count) if count > 0 else 0
                print(f"[{count:4d}/{total}] {elapsed:.0f}s | ETA {eta/60:.0f}min | "
                      f"最优收益={best_return:+.2f}%{marker}")

        except Exception as e:
            if count % 500 == 0:
                print(f"[{count:4d}/{total}] ⚠️ {e}")

    elapsed = time.time() - t0
    print(f"\n✅ 完成 {len(results)} 组 (跳过 {skipped} 组违反约束), 耗时 {elapsed/60:.1f} 分钟")

    # ==================== 排名输出 ====================
    results.sort(key=lambda x: x["return"], reverse=True)

    print(f"\n{'='*130}")
    print(f"📊 TOP 30 参数组合（按总收益率排序）")
    print(f"{'='*130}")
    h = (f"{'#':<3} {'收益%':>7} {'超额%':>7} {'回撤%':>6} {'笔':>3} "
         f"{'夏普':>6} {'稳夏':>6} {'g_dn':>5} {'g_up':>5} {'red%':>5} {'add%':>5} "
         f"{'ADX':>4} {'a_dn':>5} {'a_up':>5} {'a_rp':>5} {'m_dn':>5} {'m_up':>5} {'m_rp':>5} {'止盈':>5}")
    print(h)
    print("-" * 150)

    for i, r in enumerate(results[:30], 1):
        print(f"{i:<3} {r['return']:>7.2f} {r['excess']:>7.2f} {r['max_dd']:>6.2f} {r['trades']:>3} "
              f"{r['sharpe']:>6.2f} {r['robust_sharpe']:>6.2f} "
              f"{r['pm_grid_down']:>5.2f} {r['pm_grid_up']:>5.2f} {r['pm_reduce_pct']:>5.2f} {r['pm_add_pct']:>5.2f} "
              f"{r['pm_adx_threshold']:>4.0f} {r['pm_grid_down_agg']:>5.2f} {r['pm_grid_up_agg']:>5.2f} "
              f"{r['pm_reduce_pct_agg']:>5.2f} {r['pm_take_profit_ratio']:>5.2f}")

    # ==================== 最优输出 ====================
    print(f"\n{'='*60}")
    print("🏆 最优参数组合")
    print(f"{'='*60}")
    gd, gu, rp, ap, adx, gda, gua, rpa, tp = best_result
    gdm = round((gda + gd) / 2, 3)
    gum = round((gua + gu) / 2, 3)
    rpm = round((rp + rpa) / 2, 3)
    print(f"   pm_grid_down       = {gd:.2f}")
    print(f"   pm_grid_up         = {gu:.2f}")
    print(f"   pm_reduce_pct      = {rp:.2f}")
    print(f"   pm_add_pct         = {ap:.2f}")
    print(f"   pm_adx_threshold   = {adx:.0f}")
    print(f"   pm_grid_down_agg   = {gda:.2f}")
    print(f"   pm_grid_up_agg     = {gua:.2f}")
    print(f"   pm_reduce_pct_agg  = {rpa:.2f}")
    print(f"   pm_take_profit_ratio = {tp:.2f}")
    print("-" * 60)
    print(f"   买入持有收益率  = {buy_hold_return:+.2f}%")
    print(f"   策略总收益率    = {best_return:.2f}%")
    print(f"   超额收益       = {best_return - buy_hold_return:+.2f}%")
    if best_stats:
        print(f"   总交易次数     = {best_stats['总交易']}")
        print(f"   最大回撤       = {best_stats['最大回撤 (%)']}%")
        print(f"   最终资金       = ¥{best_stats['最终资金 (¥)']:,.2f}")
        print(f"   夏普比率       = {best_stats['夏普比率']}")
        print(f"   CAGR           = {best_stats['CAGR (%)']}%")
        print(f"   稳健夏普比率   = {best_stats['稳健夏普比率']}")
        print(f"   R立方          = {best_stats['R立方']}")
    print(f"{'='*60}")

    # 保存 Excel
    import pandas as pd
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "result")
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(results).to_excel(os.path.join(out_dir, "pm1_grid_search_results.xlsx"), index=False)
    print(f"\n📁 完整结果: result/pm1_grid_search_results.xlsx")


if __name__ == "__main__":
    grid_search()
