#!/usr/bin/env python3
"""Position Management 2 (Swing恢复) 参数网格搜索

搜索 pm2_bounce_up, pm2_buyback_down, pm2_trade_lots，
约束 bounce_up > buyback_down（否则 swing 无利润空间），
找总收益率最大的参数组合。
"""

import sys
import os
import itertools
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.config import BacktestConfig
from core.indicators import compute_indicators
from core.engine import run_backtest
from core.statistics import calc_stats


def grid_search_pm2(ts_code: str, name: str, entry_date: str,
                    initial_shares: int, initial_capital: float,
                    per_shares: int, market: str = "HK",
                    end_date: str = "20260624"):
    """对一只标的做 PM2 参数网格搜索"""
    from core.cache import load as cache_load

    df = cache_load(ts_code, market)
    df = df[(df["Date"] >= entry_date) & (df["Date"] <= end_date)].copy()
    df = df.sort_values("Date").reset_index(drop=True)
    df = compute_indicators(df, mode="position_management_2")

    current_price = float(df["Close"].iloc[-1])
    start_price = float(df["Close"].iloc[0])
    init_equity = initial_capital + initial_shares * start_price * per_shares
    bh_final = initial_capital + initial_shares * current_price * per_shares
    bh_return = (bh_final / init_equity - 1) * 100

    print(f"\n{'='*70}")
    print(f"🔍 {name} ({ts_code})  PM2 参数网格搜索")
    print(f"   区间: {entry_date} → {end_date}  |  "
          f"¥{start_price:.0f}→¥{current_price:.0f}  死扛: {bh_return:+.2f}%")
    print(f"   初始{initial_shares}手  现金¥{initial_capital:,}  初始权益¥{init_equity:,.0f}")
    print(f"{'='*70}")

    # 搜索空间（约束 bounce_up > buyback_down）
    bounce_ups = [0.05, 0.08, 0.10, 0.12, 0.15]
    buyback_downs = [0.03, 0.05, 0.08, 0.10]
    trade_lots_list = [1, 2, 3, 5]

    # 预过滤：bounce_up > buyback_down
    combinations = [(b, bb, tl)
                    for b in bounce_ups
                    for bb in buyback_downs
                    for tl in trade_lots_list
                    if b > bb]

    best_return = -float("inf")
    best_params = None
    best_stats = None
    all_results = []

    total = len(combinations)
    for count, (bounce, buyback, lots) in enumerate(combinations, 1):
        try:
            config = BacktestConfig(
                ts_code=ts_code, stock_name=name, market=market,
                mode="position_management_2",
                pm_start_date=entry_date,
                pm_initial_shares=initial_shares,
                initial_capital=initial_capital,
                per_shares=per_shares,
                data_source="akshare",
                pm2_bounce_up=bounce,
                pm2_buyback_down=buyback,
                pm2_trade_lots=lots,
            )
            trades, equity = run_backtest(df.copy(), config)
            stats = calc_stats(trades, config.initial_capital, equity)
            ret = stats["总收益率 (%)"]

            all_results.append({
                "bounce": bounce, "buyback": buyback, "lots": lots,
                "return": ret, "trades": stats["总交易"],
                "max_dd": stats["最大回撤 (%)"],
                "final_equity": stats["最终资金 (¥)"],
            })

            marker = " ⭐" if ret > best_return else ""
            if ret > best_return:
                best_return = ret
                best_params = (bounce, buyback, lots)
                best_stats = stats

            print(f"  [{count:3d}/{total}] "
                  f"反弹+{bounce*100:4.0f}% 回落{buyback*100:3.0f}%  {lots}手 "
                  f"→ 收益={ret:+8.2f}%  交易={stats['总交易']:3d}  "
                  f"回撤={stats['最大回撤 (%)']:+6.1f}%{marker}")

        except Exception as e:
            print(f"  [{count:3d}/{total}] "
                  f"反弹+{bounce*100:4.0f}% 回落{buyback*100:3.0f}%  {lots}手 "
                  f"→ ❌ {e}")

    # ─── 汇总 ───
    sorted_results = sorted(all_results, key=lambda x: x["return"], reverse=True)

    print(f"\n  📊 Top 15 参数组合（按收益率排序）：")
    print(f"  {'排名':<4} {'反弹':>6} {'回落':>6} {'手数':>4} "
          f"{'收益率':>10} {'vs死扛':>10} {'交易':>5} {'回撤':>8} {'最终资金':>14}")
    print(f"  {'-'*72}")
    for i, r in enumerate(sorted_results[:15]):
        diff = r["return"] - bh_return
        print(f"  {i+1:<4} {r['bounce']*100:>5.0f}% {r['buyback']*100:>5.0f}% "
              f"{r['lots']:>4}手 {r['return']:>+9.2f}% {diff:>+9.2f}% "
              f"{r['trades']:>4}  {r['max_dd']:>+7.1f}% ¥{r['final_equity']:>12,.0f}")

    # ─── 最优 ───
    r = sorted_results[0]
    print(f"\n  🏆 最优参数: 反弹+{r['bounce']*100:.0f}% → 回落{r['buyback']*100:.0f}%买回  每次{r['lots']}手")
    print(f"     收益率: {r['return']:+.2f}%  (死扛 {bh_return:+.2f}%，改善 {r['return']-bh_return:+.2f}%)")
    print(f"     交易: {r['trades']}次  最终资金: ¥{r['final_equity']:,.0f}  最大回撤: {r['max_dd']:+.1f}%")

    return sorted_results[0]


if __name__ == "__main__":
    print("=" * 70)
    print("🔍 Position Management 2 (Swing恢复) — 参数网格搜索")
    print("=" * 70)

    # 理想汽车：从2025-10-01开始（当时约¥80，已从高点回落）
    # grid_search_pm2(
    #     ts_code="02015.HK", name="理想汽车",
    #     entry_date="20251021",
    #     initial_shares=25, initial_capital=40000, per_shares=100, market="HK",
    # )

    # 老铺黄金
    grid_search_pm2(
        ts_code="06181.HK", name="老铺黄金",
        entry_date="20260424",
        initial_shares=3, initial_capital=40000, per_shares=100, market="HK",
    )
