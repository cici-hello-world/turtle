#!/usr/bin/env python3
"""深套场景：利用持仓做 swing 交易，靠波动差价摊薄成本

策略逻辑：
  1. 从当前持仓出发（已亏损，成本价=入场价）
  2. 价格从近期低点反弹 X% → 卖出 Y 手
  3. 价格从卖出价回落 Z% → 买回 Y 手（或更多）
  4. 反复操作，差价利润用于降低有效成本
  5. 目标：即使股价不回到入场价，通过交易利润也能回本

对比：
  - 死扛：不动，等股价回来
  - 反弹减仓：只卖不买，降低仓位锁定亏损
  - Swing恢复：高卖低买，用波动赚钱摊薄成本
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_swing_recovery(df: pd.DataFrame, initial_shares: int,
                       initial_capital: float, entry_price: float,
                       per_shares: int, bounce_pct: float,
                       buyback_pct: float, trade_lots: int,
                       market: str = "HK") -> dict:
    """Swing 恢复策略

    Args:
        bounce_pct: 从近期低点反弹多少触发卖出
        buyback_pct: 从卖出价回落多少触发买回
        trade_lots: 每次交易手数
    Returns:
        dict with results including effective cost basis
    """
    trades = []
    capital = initial_capital
    lots = initial_shares
    recent_low = float(df["Close"].iloc[0])  # 跟踪近期低点

    # 成本追踪
    total_cost_basis = initial_shares * entry_price * per_shares  # 初始总成本
    last_sell_price = 0  # 最近一次卖出价，用于判断买回

    state = "holding"  # holding | waiting_buyback
    equity_curve = [capital + lots * recent_low * per_shares]

    for i in range(1, len(df)):
        close = float(df["Close"].iloc[i])
        current_date = df["Date"].iloc[i]

        if close < recent_low:
            recent_low = close

        if state == "holding":
            # 反弹触发卖出
            bounce_trigger = recent_low * (1 + bounce_pct)
            if close >= bounce_trigger and lots >= trade_lots:
                sell_lots = min(trade_lots, lots)
                gross = close * sell_lots * per_shares
                cost = gross * (0.0012 if market == "HK" else 0.0006)
                net = gross - cost
                capital += net
                lots -= sell_lots

                # 从总成本中扣除卖出部分（按比例）
                cost_per_lot = total_cost_basis / (lots + sell_lots)
                total_cost_basis -= cost_per_lot * sell_lots

                trades.append({
                    "Date": current_date, "Type": "卖出",
                    "Price": close, "Lots": sell_lots,
                    "Net": net, "Lots_After": lots,
                    "Capital": capital,
                    "Cost_Basis_Remaining": total_cost_basis,
                })
                last_sell_price = close
                recent_low = close  # 重置低点
                state = "waiting_buyback"

        elif state == "waiting_buyback":
            # 回落触发买回
            buyback_trigger = last_sell_price * (1 - buyback_pct)
            if close <= buyback_trigger:
                buy_lots = trade_lots
                gross = close * buy_lots * per_shares
                cost = gross * (0.0012 if market == "HK" else 0.0006)
                total_cost_spent = gross + cost
                if total_cost_spent <= capital:
                    capital -= total_cost_spent
                    lots += buy_lots
                    total_cost_basis += gross  # 买回成本加入

                    trades.append({
                        "Date": current_date, "Type": "买回",
                        "Price": close, "Lots": buy_lots,
                        "Net": -total_cost_spent, "Lots_After": lots,
                        "Capital": capital,
                        "Cost_Basis_Remaining": total_cost_basis,
                    })
                    recent_low = close
                    state = "holding"
            else:
                # 还在等买回，但价格创新低 → 重置状态
                if close < recent_low:
                    recent_low = close

        equity = capital + lots * close * per_shares
        equity_curve.append(equity)

    # 最终统计
    final_price = float(df["Close"].iloc[-1])
    final_equity = capital + lots * final_price * per_shares
    avg_cost_per_lot = total_cost_basis / lots if lots > 0 else 0
    avg_cost_per_share = avg_cost_per_lot / per_shares if lots > 0 else 0

    # swing 利润 = 交易净现金流
    swing_profit = sum(t["Net"] for t in trades)

    return {
        "trades": trades,
        "final_lots": lots,
        "final_capital": capital,
        "final_equity": final_equity,
        "total_cost_basis": total_cost_basis,
        "avg_cost_per_share": avg_cost_per_share,
        "swing_profit": swing_profit,
        "entry_price": entry_price,
        "final_price": final_price,
        "equity_curve": equity_curve,
    }


def analyze_swing(ts_code: str, name: str,
                  entry_price: float, initial_shares: int,
                  initial_capital: float, per_shares: int,
                  start_date: str, end_date: str = "20260624"):
    """分析 swing 恢复策略"""
    from core.cache import load as cache_load

    df = cache_load(ts_code, "HK")
    df = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)].copy()
    df = df.sort_values("Date").reset_index(drop=True)

    current_price = float(df["Close"].iloc[-1])
    init_equity = initial_capital + initial_shares * entry_price * per_shares

    # 死扛基准
    bh_final = initial_capital + initial_shares * current_price * per_shares
    bh_return = (bh_final / init_equity - 1) * 100

    print(f"\n{'='*70}")
    print(f"📊 {name} ({ts_code}) — Swing 恢复策略")
    print(f"   区间: {start_date} → {end_date}")
    print(f"   入场¥{entry_price:.0f} → 现价¥{current_price:.0f}  死扛: {bh_return:+.1f}%")
    print(f"   初始{initial_shares}手  现金¥{initial_capital:,}  初始权益¥{init_equity:,.0f}")
    print(f"{'='*70}")

    # 搜索参数
    bounce_pcts = [0.05, 0.08, 0.10, 0.12]
    buyback_pcts = [0.03, 0.05, 0.08, 0.10]
    trade_lots_list = [1, 2, 3]

    best_equity = -float("inf")
    best_result = None
    best_params = None
    all_results = []

    for bounce, buyback, tl in [
        (b, bb, tl) for b in bounce_pcts for bb in buyback_pcts for tl in trade_lots_list
        if bb < b  # 买回跌幅必须小于反弹幅度，否则亏差价
    ]:
        result = run_swing_recovery(
            df.copy(), initial_shares, initial_capital, entry_price,
            per_shares, bounce_pct=bounce, buyback_pct=buyback,
            trade_lots=tl, market="HK",
        )
        final_eq = result["final_equity"]
        ret = (final_eq / init_equity - 1) * 100

        # 计算降低的成本
        cost_reduction = (1 - result["avg_cost_per_share"] / entry_price) * 100 if result["avg_cost_per_share"] > 0 else 0
        breakeven_price = result["avg_cost_per_share"] if result["final_lots"] > 0 else 0

        all_results.append({
            "bounce": bounce, "buyback": buyback, "trade_lots": tl,
            "return": ret, "equity": final_eq,
            "swing_profit": result["swing_profit"],
            "trades_n": len(result["trades"]),
            "final_lots": result["final_lots"],
            "avg_cost": result["avg_cost_per_share"],
            "breakeven": breakeven_price,
            "cost_reduction": cost_reduction,
        })

        if ret > best_equity:
            best_equity = ret
            best_result = result
            best_params = (bounce, buyback, tl)

    # 排序输出
    sorted_results = sorted(all_results, key=lambda x: x["return"], reverse=True)

    print(f"\n  📊 Top 15 参数组合：")
    print(f"  {'排名':<4} {'反弹卖':>7} {'回落买':>7} {'手数':>4} "
          f"{'收益率':>9} {'vs死扛':>8} {'Swing利润':>10} "
          f"{'交易':>4} {'剩余':>4} {'成本→':>9} {'回本价':>9}")
    print(f"  {'-'*82}")
    for i, r in enumerate(sorted_results[:15]):
        diff = r["return"] - bh_return
        print(f"  {i+1:<4} {r['bounce']*100:>6.0f}% {r['buyback']*100:>6.0f}% "
              f"{r['trade_lots']:>4}手 "
              f"{r['return']:>+8.2f}% {diff:>+7.2f}% "
              f"¥{r['swing_profit']:>+8,.0f} {r['trades_n']:>4} {r['final_lots']:>4}手 "
              f"¥{r['avg_cost']:>7.0f} ¥{r['breakeven']:>7.0f}")

    # 最优详情
    r = sorted_results[0]
    print(f"\n  🏆 最优参数: 反弹+{r['bounce']*100:.0f}%卖 → 回落{r['buyback']*100:.0f}%买回  每次{r['trade_lots']}手")
    print(f"     收益率: {r['return']:+.2f}%  (死扛 {bh_return:+.2f}%)")
    print(f"     Swing利润: ¥{r['swing_profit']:+,.0f}  交易{r['trades_n']}次")
    print(f"     剩余仓位: {r['final_lots']}手  ")
    print(f"     原始成本: ¥{entry_price:.0f}/股 → 当前成本: ¥{r['avg_cost']:.0f}/股 "
          f"(降低 {r['cost_reduction']:.1f}%)")
    print(f"     回本所需价: ¥{r['breakeven']:.0f}  (当前¥{current_price:.0f}，"
          f"还需涨 {(r['breakeven']/current_price-1)*100:.0f}%)")

    # 死扛回本对比
    print(f"     死扛回本价: ¥{entry_price:.0f}  (需涨 {(entry_price/current_price-1)*100:.0f}%)")

    return best_params, best_result


if __name__ == "__main__":
    print("=" * 70)
    print("🔍 Swing恢复策略 — 用波动差价摊薄成本")
    print("=" * 70)

    # ─── 理想汽车：从今天开始（用最近半年数据模拟类似处境） ───
    # 从2025-10月开始，当时股价≈¥80（距高点已跌36%，类似"已经深套"的状态）
    analyze_swing(
        ts_code="02015.HK", name="理想汽车",
        entry_price=100.0, initial_shares=26, initial_capital=40000,
        per_shares=100, start_date="20251001",
    )

    # ─── 老铺黄金 ───
    analyze_swing(
        ts_code="06181.HK", name="老铺黄金",
        entry_price=650.0, initial_shares=3, initial_capital=40000,
        per_shares=100, start_date="20251001",
    )
