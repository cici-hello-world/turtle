#!/usr/bin/env python3
"""深套场景下多种策略对比回测

对比策略：
  1. 死扛 (Buy & Hold)          — 基准
  2. 反马丁 (Anti-Martingale)    — MA信号入场，有止损，亏损缩仓
  3. 反弹减仓 (Bounce Reduce)    — 每次反弹N%卖出M%仓位，越涨越卖
  4. 均线止损 (MA Stop)          — 价格<MA时空仓，>MA时持仓
  5. 仓位网格 (PM1 Grid)         — 之前测试的网格策略（对照）

所有策略从同一时间点启动，跑完完整回测区间，对比最终收益。
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.config import BacktestConfig
from core.indicators import compute_indicators
from core.engine import run_backtest
from core.statistics import calc_stats


# ==================== 反弹减仓策略 ====================

def run_bounce_reduce(df: pd.DataFrame, initial_shares: int,
                      initial_capital: float, per_shares: int,
                      bounce_pct: float, reduce_frac: float,
                      market: str = "HK") -> tuple[list[dict], list[float]]:
    """反弹减仓：价格从近期低点反弹 bounce_pct 时，卖出 reduce_frac 比例仓位

    核心思路：不接飞刀，只在反弹时减仓。反弹越多卖越多。
    每次卖出后，重置"近期低点"为卖出价。
    """
    trades = []
    capital = initial_capital
    lots = initial_shares
    recent_low = float(df["Close"].iloc[0])  # 跟踪近期低点
    equity_curve = [capital + lots * recent_low * per_shares]

    for i in range(1, len(df)):
        close = float(df["Close"].iloc[i])
        current_date = df["Date"].iloc[i]

        # 更新近期低点
        if close < recent_low:
            recent_low = close

        # 反弹触发卖出？
        bounce_trigger = recent_low * (1 + bounce_pct)
        if close >= bounce_trigger and lots > 0:
            # 按比例卖出
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
                "Cost_¥": cost,
                "Capital_After": capital,
                "Reason": f"反弹减仓 +{int(bounce_pct*100)}%（卖{int(reduce_frac*100)}%）",
                "Amount": net, "Position_After": lots,
                "Equity_After": capital + lots * close * per_shares,
                "Reference_Price": close,
            })

            # 重置低点为卖出价
            recent_low = close

        equity = capital + lots * close * per_shares
        equity_curve.append(equity)

    return trades, equity_curve


# ==================== 均线止损策略 ====================

def run_ma_stop(df: pd.DataFrame, initial_shares: int,
                initial_capital: float, per_shares: int,
                ma_period: int, market: str = "HK") -> tuple[list[dict], list[float]]:
    """均线止损：Close < MA(ma_period) 时空仓（卖出所有），Close > MA 时满仓（买入）

    只在穿越时交易一次，避免反复摩擦。
    """
    ma = df["Close"].rolling(ma_period).mean()
    trades = []
    capital = initial_capital
    lots = initial_shares
    in_position = True  # 初始持仓
    equity_curve = [capital + lots * float(df["Close"].iloc[0]) * per_shares]

    for i in range(ma_period, len(df)):
        close = float(df["Close"].iloc[i])
        current_date = df["Date"].iloc[i]
        ma_val = float(ma.iloc[i])
        if pd.isna(ma_val):
            continue

        if in_position and close < ma_val:
            # 跌破均线 → 全部卖出
            if lots > 0:
                gross = close * lots * per_shares
                cost = gross * (0.0012 if market == "HK" else 0.0006)
                net = gross - cost
                capital += net
                trades.append({
                    "Date": current_date, "Entry Date": current_date,
                    "Exit Date": current_date, "Type": "清仓",
                    "Price": close, "Entry Price": close, "Exit Price": close,
                    "Shares": lots, "PnL_¥": net, "PnL_%": 0.0,
                    "Cost_¥": cost,
                    "Reason": f"跌破MA{ma_period}",
                    "Capital_After": capital, "Position_After": 0,
                    "Amount": net, "Equity_After": capital,
                    "Reference_Price": close,
                })
                lots = 0
            in_position = False

        elif not in_position and close > ma_val:
            # 站上均线 → 全部买入
            buy_lots = int(capital / (close * per_shares * 1.002))
            if buy_lots > 0:
                gross = close * buy_lots * per_shares
                cost = gross * (0.0012 if market == "HK" else 0.0006)
                total = gross + cost
                capital -= total
                lots = buy_lots
                equity = capital + lots * close * per_shares
                trades.append({
                    "Date": current_date, "Entry Date": current_date,
                    "Exit Date": current_date, "Type": "满仓",
                    "Price": close, "Entry Price": close, "Exit Price": close,
                    "Shares": buy_lots, "PnL_¥": -total, "PnL_%": 0.0,
                    "Cost_¥": cost,
                    "Reason": f"站上MA{ma_period}",
                    "Capital_After": capital, "Position_After": lots,
                    "Amount": -total, "Equity_After": equity,
                    "Reference_Price": close,
                })
            in_position = True

        equity = capital + lots * close * per_shares
        equity_curve.append(equity)

    return trades, equity_curve


# ==================== 主流程 ====================

def compare_strategies(ts_code: str, name: str,
                       entry_date: str, entry_price: float,
                       initial_shares: int, initial_capital: float,
                       per_shares: int, market: str = "HK",
                       end_date: str = "20260624"):
    """比较多种策略"""
    from core.cache import load as cache_load

    df = cache_load(ts_code, market)
    if df is None or df.empty:
        print(f"❌ {name}: 无数据")
        return

    df = df[(df["Date"] >= entry_date) & (df["Date"] <= end_date)].copy()
    df = df.sort_values("Date").reset_index(drop=True)
    if len(df) < 60:
        print(f"❌ {name}: 数据不足 ({len(df)}行)")
        return

    df = compute_indicators(df, mode="position_management_1")

    current_price = float(df["Close"].iloc[-1])
    current_value = initial_shares * current_price * per_shares
    cost_basis = initial_shares * entry_price * per_shares
    current_loss = (current_price / entry_price - 1) * 100

    print(f"\n{'='*70}")
    print(f"📊 {name} ({ts_code})")
    print(f"   入场价: ¥{entry_price:.2f} → 现价: ¥{current_price:.2f}  "
          f"浮亏: {current_loss:.1f}%  |  市值: ¥{current_value:,}  |  "
          f"现金: ¥{initial_capital:,}")
    print(f"   回测区间: {entry_date} → {end_date}  ({len(df)}个交易日)")
    print(f"{'='*70}")

    results = {}

    # 初始权益 = 现金 + 持仓市值
    init_equity = initial_capital + initial_shares * entry_price * per_shares

    # ─── 1. 死扛基准 ───
    bh_final = initial_capital + initial_shares * current_price * per_shares
    bh_return = (bh_final / init_equity - 1) * 100
    results["死扛"] = {"return": bh_return, "final": bh_final, "trades": 0, "max_dd": 0}

    # ─── 2. 反马丁 (Anti-Martingale) ───
    print("\n  🧪 测试 反马丁策略 ...")
    try:
        config_am = BacktestConfig(
            ts_code=ts_code, stock_name=name, market=market,
            mode="anti_martingale",
            initial_capital=initial_capital,
            per_shares=per_shares,
            data_source="akshare",
            am_ma_period=50,
            am_risk_multiplier=1.5,
            am_max_multiplier=4.0,
            ema_fast=25, ema_slow=80,
            risk_percent=2.0,
        )
        trades_am, equity_am = run_backtest(df, config_am)
        stats_am = calc_stats(trades_am, initial_capital, equity_am)
        results["反马丁(纯现金起步)"] = {
            "return": stats_am["总收益率 (%)"],
            "final": stats_am["最终资金 (¥)"],
            "trades": stats_am["总交易"],
            "max_dd": stats_am["最大回撤 (%)"],
        }
    except Exception as e:
        print(f"    ⚠️ 反马丁失败: {e}")
        results["反马丁(纯现金起步)"] = {"return": 0, "final": initial_capital, "trades": 0, "max_dd": 0}

    # ─── 3. 反弹减仓 ───
    print("  🧪 测试 反弹减仓策略 ...")
    for bounce_pct, reduce_frac, label_suffix in [
        (0.05, 0.25, "反弹5%卖25%"),
        (0.08, 0.33, "反弹8%卖33%"),
        (0.10, 0.50, "反弹10%卖50%"),
    ]:
        try:
            df_copy = df.copy()
            trades_br, equity_br = run_bounce_reduce(
                df_copy, initial_shares, initial_capital, per_shares,
                bounce_pct=bounce_pct, reduce_frac=reduce_frac, market=market,
            )
            stats_br = calc_stats(trades_br, init_equity, equity_br)
            label = f"反弹减仓({label_suffix})"
            results[label] = {
                "return": stats_br["总收益率 (%)"],
                "final": stats_br["最终资金 (¥)"],
                "trades": stats_br["总交易"],
                "max_dd": stats_br["最大回撤 (%)"],
            }
        except Exception as e:
            print(f"    ⚠️ {label_suffix} 失败: {e}")

    # ─── 4. 均线止损 ───
    print("  🧪 测试 均线止损策略 ...")
    for ma_period in [20, 50, 60]:
        try:
            df_copy = df.copy()
            trades_ma, equity_ma = run_ma_stop(
                df_copy, initial_shares, initial_capital, per_shares,
                ma_period=ma_period, market=market,
            )
            if not trades_ma:
                print(f"    ⚠️ MA{ma_period}: 无交易信号")
                # 无交易 → 等价于死扛
                results[f"MA{ma_period}止损"] = {
                    "return": bh_return, "final": bh_final, "trades": 0, "max_dd": 0,
                }
                continue
            stats_ma = calc_stats(trades_ma, init_equity, equity_ma)
            label = f"MA{ma_period}止损"
            results[label] = {
                "return": stats_ma["总收益率 (%)"],
                "final": stats_ma["最终资金 (¥)"],
                "trades": stats_ma["总交易"],
                "max_dd": stats_ma["最大回撤 (%)"],
            }
        except Exception as e:
            print(f"    ⚠️ MA{ma_period} 异常: {e}")
            import traceback; traceback.print_exc()

    # ─── 5. 仓位网格(对照) ───
    print("  🧪 测试 仓位网格策略 ...")
    try:
        config_pm = BacktestConfig(
            ts_code=ts_code, stock_name=name, market=market,
            mode="position_management_1",
            pm_start_date=entry_date,
            pm_initial_shares=initial_shares,
            initial_capital=initial_capital,
            per_shares=per_shares,
            data_source="akshare",
            pm_grid_up=0.03, pm_grid_down=0.03,
            pm_add_pct=0.03, pm_reduce_pct=0.75,
        )
        trades_pm, equity_pm = run_backtest(df, config_pm)
        stats_pm = calc_stats(trades_pm, initial_capital, equity_pm)
        results["仓位网格(PM1)"] = {
            "return": stats_pm["总收益率 (%)"],
            "final": stats_pm["最终资金 (¥)"],
            "trades": stats_pm["总交易"],
            "max_dd": stats_pm["最大回撤 (%)"],
        }
    except Exception as e:
        print(f"    ⚠️ 仓位网格失败: {e}")
        results["仓位网格(PM1)"] = {"return": 0, "final": 0, "trades": 0, "max_dd": 0}

    # ─── 汇总 ───
    print(f"\n  📊 策略对比汇总：")
    print(f"  {'策略':<22} {'收益率':>10} {'最终资金':>14} {'最大回撤':>10} {'交易次数':>8}")
    print(f"  {'-'*65}")

    # 含仓策略（可比较）
    position_strategies = {k: v for k, v in results.items() if "纯现金" not in k}
    sorted_pos = sorted(position_strategies.items(), key=lambda x: x[1]["return"], reverse=True)
    for label, r in sorted_pos:
        print(f"  {label:<22} {r['return']:>+9.2f}% ¥{r['final']:>12,.0f} "
              f"{r['max_dd']:>+9.2f}% {r['trades']:>7}")
    # 纯现金策略单独列出
    cash_strategies = {k: v for k, v in results.items() if "纯现金" in k}
    if cash_strategies:
        print(f"  {'─'*65}")
        print(f"  [纯现金起步，不持有初始仓位]")
        for label, r in cash_strategies.items():
            print(f"  {label:<22} {r['return']:>+9.2f}% ¥{r['final']:>12,.0f} "
                  f"{r['max_dd']:>+9.2f}% {r['trades']:>7}")
    print(f"  {'-'*65}")

    if sorted_pos:
        best_label, best_r = sorted_pos[0]
        print(f"  🏆 最优(含仓): {best_label}  死扛={results['死扛']['return']:+.2f}%  "
              f"差额={best_r['return'] - results['死扛']['return']:+.2f}%")


# ==================== main ====================
if __name__ == "__main__":
    print("=" * 70)
    print("🔍 深套持仓 — 多策略对比回测")
    print("=" * 70)

    # ─── 理想汽车 ───
    # 假设入场价 ¥100（具体请根据实际调整）
    compare_strategies(
        ts_code="02015.HK", name="理想汽车",
        entry_date="20241001", entry_price=100.0,
        initial_shares=26, initial_capital=40000,
        per_shares=100, market="HK",
    )

    # ─── 老铺黄金 ───
    # 假设入场价 ¥650（具体请根据实际调整）
    compare_strategies(
        ts_code="06181.HK", name="老铺黄金",
        entry_date="20250601", entry_price=650.0,
        initial_shares=3, initial_capital=40000,
        per_shares=100, market="HK",
    )
