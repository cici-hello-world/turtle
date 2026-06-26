#!/usr/bin/env python3
"""反弹减仓策略参数网格搜索

搜索 bounce_pct（反弹幅度）和 reduce_frac（卖出比例）的最优组合，
目标：总收益率最大化。
"""

import sys
import os
import itertools
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.statistics import calc_stats


def run_bounce_reduce(df: pd.DataFrame, initial_shares: int,
                      initial_capital: float, per_shares: int,
                      bounce_pct: float, reduce_frac: float,
                      market: str = "HK") -> tuple[list[dict], list[float]]:
    """反弹减仓：价格从近期低点反弹 bounce_pct 时，卖出 reduce_frac 比例仓位"""
    trades = []
    capital = initial_capital
    lots = initial_shares
    recent_low = float(df["Close"].iloc[0])
    equity_curve = [capital + lots * recent_low * per_shares]

    for i in range(1, len(df)):
        close = float(df["Close"].iloc[i])
        current_date = df["Date"].iloc[i]

        if close < recent_low:
            recent_low = close

        bounce_trigger = recent_low * (1 + bounce_pct)
        if close >= bounce_trigger and lots > 0:
            sell_lots = max(1, int(lots * reduce_frac))
            sell_lots = min(sell_lots, lots)

            gross = close * sell_lots * per_shares
            cost = gross * (0.0012 if market == "HK" else 0.0006)
            net = gross - cost
            capital += net
            lots -= sell_lots

            trades.append({
                "Date": current_date, "Entry Date": current_date,
                "Exit Date": current_date, "Type": "减仓",
                "Price": close, "Entry Price": close, "Exit Price": close,
                "Shares": sell_lots, "PnL_¥": net, "PnL_%": 0.0,
                "Cost_¥": cost, "Capital_After": capital,
                "Reason": f"反弹+{int(bounce_pct*100)}%卖{int(reduce_frac*100)}%",
                "Amount": net, "Position_After": lots,
                "Equity_After": capital + lots * close * per_shares,
                "Reference_Price": close,
            })
            recent_low = close

        equity_curve.append(capital + lots * close * per_shares)

    return trades, equity_curve


def grid_search_bounce(ts_code: str, name: str, entry_date: str,
                       entry_price: float, initial_shares: int,
                       initial_capital: float, per_shares: int,
                       market: str = "HK", end_date: str = "20260624"):
    """对一只标的做反弹参数网格搜索"""
    from core.cache import load as cache_load

    df = cache_load(ts_code, market)
    df = df[(df["Date"] >= entry_date) & (df["Date"] <= end_date)].copy()
    df = df.sort_values("Date").reset_index(drop=True)
    current_price = float(df["Close"].iloc[-1])

    init_equity = initial_capital + initial_shares * entry_price * per_shares
    bh_final = initial_capital + initial_shares * current_price * per_shares
    bh_return = (bh_final / init_equity - 1) * 100

    print(f"\n{'='*70}")
    print(f"🔍 {name} ({ts_code})  反弹减仓参数网格搜索")
    print(f"   入场¥{entry_price:.0f}→现价¥{current_price:.0f}  死扛收益: {bh_return:+.2f}%")
    print(f"   初始仓位{initial_shares}手  现金¥{initial_capital:,}  初始权益¥{init_equity:,.0f}")
    print(f"{'='*70}")

    # 搜索空间
    bounce_pcts = [0.03, 0.05, 0.07, 0.08, 0.10, 0.12, 0.15]
    reduce_fracs = [0.15, 0.20, 0.25, 0.30, 0.33, 0.40, 0.50]

    total = len(bounce_pcts) * len(reduce_fracs)
    best_return = -float("inf")
    best_params = None
    best_stats = None
    best_trades = None
    all_results = []

    count = 0
    for bounce, reduce_f in itertools.product(bounce_pcts, reduce_fracs):
        count += 1
        try:
            trades, equity = run_bounce_reduce(
                df.copy(), initial_shares, initial_capital, per_shares,
                bounce_pct=bounce, reduce_frac=reduce_f, market=market,
            )
            stats = calc_stats(trades, init_equity, equity)
            ret = stats["总收益率 (%)"]
            all_results.append((bounce, reduce_f, ret, stats["总交易"], stats["最大回撤 (%)"]))

            marker = " ⭐" if ret > best_return else ""
            if ret > best_return:
                best_return = ret
                best_params = (bounce, reduce_f)
                best_stats = stats
                best_trades = trades

            print(f"  [{count:3d}/{total}] "
                  f"反弹+{bounce*100:4.0f}% 卖{reduce_f*100:3.0f}% "
                  f"→ 收益={ret:+.2f}%  交易={stats['总交易']:2d}  "
                  f"回撤={stats['最大回撤 (%)']:+.1f}%{marker}")

        except Exception as e:
            print(f"  [{count:3d}/{total}] "
                  f"反弹+{bounce*100:4.0f}% 卖{reduce_f*100:3.0f}% → ❌ {e}")

    # 汇总
    print(f"\n  📊 Top 10 参数组合：")
    print(f"  {'排名':<5} {'反弹幅度':>8} {'卖出比例':>8} {'收益率':>10} {'vs死扛':>10} {'交易':>5} {'回撤':>8}")
    print(f"  {'-'*58}")
    sorted_results = sorted(all_results, key=lambda x: x[2], reverse=True)
    for i, (b, r, ret, trades_n, dd) in enumerate(sorted_results[:10]):
        diff = ret - bh_return
        print(f"  {i+1:<5} {b*100:>7.0f}% {r*100:>7.0f}% "
              f"{ret:>+9.2f}% {diff:>+9.2f}% {trades_n:>4}  {dd:>+7.1f}%")

    # 最优详情
    print(f"\n  🏆 最优参数: 反弹+{best_params[0]*100:.0f}% 卖{best_params[1]*100:.0f}%")
    print(f"     收益率: {best_return:+.2f}%  (死扛 {bh_return:+.2f}%, 改善 {best_return-bh_return:+.2f}%)")
    if best_stats:
        print(f"     交易: {best_stats['总交易']}次  最终资金: ¥{best_stats['最终资金 (¥)']:,.0f}  "
              f"最大回撤: {best_stats['最大回撤 (%)']:+.1f}%")

        # 交易摘要
        if best_trades:
            sell_prices = [t["Price"] for t in best_trades]
            print(f"     卖出均价: ¥{np.mean(sell_prices):.2f}  "
                  f"剩余仓位: {best_trades[-1]['Position_After']}手")

    return best_params, best_return


if __name__ == "__main__":
    print("=" * 70)
    print("🔍 反弹减仓策略 — 参数网格搜索")
    print("=" * 70)

    # 理想汽车
    grid_search_bounce(
        ts_code="02015.HK", name="理想汽车",
        entry_date="20241001", entry_price=100.0,
        initial_shares=26, initial_capital=40000, per_shares=100, market="HK",
    )

    # 老铺黄金
    grid_search_bounce(
        ts_code="06181.HK", name="老铺黄金",
        entry_date="20250601", entry_price=650.0,
        initial_shares=3, initial_capital=40000, per_shares=100, market="HK",
    )
