#!/usr/bin/env python3
"""针对深套持仓的仓位管理策略可行性回测

完全独立的分析脚本，不依赖 simulate_market_scenarios.py 的全局参数，
所有策略参数显式传入，避免参数串扰。
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

# ==================== K线模拟器（内联，不依赖外部脚本的全局变量） ====================

def generate_ohlcv(start_date: str, days: int, start_price: float,
                   trend_type: str) -> pd.DataFrame:
    """生成模拟日线 OHLCV 数据"""
    seed_map = {
        "up": 42, "down": 43,
        "volatile_up": 44, "volatile_down": 45,
        "sideways": 46, "v_reversal": 47,
        "inv_v_reversal": 48, "slow_grind_down": 49,
    }
    np.random.seed(seed_map[trend_type])
    dates = pd.bdate_range(start=start_date, periods=days)
    n = len(dates)
    t = np.arange(n)

    if trend_type == "up":
        trend_slope, noise_std = 0.008, 0.012
        osc_amp, osc_period = 0.0, 60
    elif trend_type == "down":
        trend_slope, noise_std = -0.008, 0.012
        osc_amp, osc_period = 0.0, 60
    elif trend_type == "volatile_up":
        trend_slope, noise_std = 0.003, 0.016
        osc_amp, osc_period = 0.06, 14
    elif trend_type == "volatile_down":
        trend_slope, noise_std = -0.002, 0.017
        osc_amp, osc_period = 0.08, 12
    elif trend_type == "sideways":
        trend_slope, noise_std = 0.0, 0.014
        osc_amp, osc_period = 0.06, 20
    elif trend_type == "v_reversal":
        trend_slope, noise_std = 0.0, 0.014
        osc_amp, osc_period = 0.0, 60
    elif trend_type == "inv_v_reversal":
        trend_slope, noise_std = 0.0, 0.014
        osc_amp, osc_period = 0.0, 60
    elif trend_type == "slow_grind_down":
        trend_slope, noise_std = -0.0035, 0.012
        osc_amp, osc_period = 0.03, 10
    else:
        raise ValueError(f"未知趋势类型: {trend_type}")

    if trend_type in ("v_reversal", "inv_v_reversal"):
        half = n // 2
        path = np.zeros(n)
        if trend_type == "v_reversal":
            path[:half] = -0.012 * np.arange(half)
            path[half:] = -0.012 * half + 0.012 * np.arange(n - half)
        else:
            path[:half] = 0.012 * np.arange(half)
            path[half:] = 0.012 * half - 0.012 * np.arange(n - half)
        noise = np.random.normal(0, noise_std, n)
        daily_log_ret = np.zeros(n)
        daily_log_ret[1:] = path[1:] - path[:-1]
        daily_log_ret[0] = path[0]
        daily_log_ret += noise
    else:
        trend = trend_slope * t
        oscillation = osc_amp * np.sin(2 * np.pi * t / osc_period)
        daily_drift = np.zeros(n); daily_drift[1:] = trend[1:] - trend[:-1]; daily_drift[0] = trend[0]
        daily_osc = np.zeros(n); daily_osc[1:] = oscillation[1:] - oscillation[:-1]; daily_osc[0] = oscillation[0]
        noise = np.random.normal(0, noise_std, n)
        daily_log_ret = daily_drift + daily_osc + noise

    close_series = start_price * np.exp(np.cumsum(daily_log_ret))
    daily_volatility = np.abs(daily_log_ret) + 0.01

    opens, highs, lows = np.zeros(n), np.zeros(n), np.zeros(n)
    for i in range(n):
        prev_close = start_price if i == 0 else close_series[i - 1]
        c = close_series[i]; vol = daily_volatility[i]
        o = prev_close * (1 + np.random.normal(0, vol * 0.3))
        h = max(o, c) + abs(np.random.normal(0, vol * 0.5))
        l = min(o, c) - abs(np.random.normal(0, vol * 0.5))
        if l <= 0: l = min(o, c) * 0.98
        opens[i], highs[i], lows[i] = o, h, l

    avg_vol = 50000
    volumes = np.random.lognormal(np.log(avg_vol), 0.4, n).astype(int)
    amounts = (volumes * close_series * 100 / 1000).round(2)

    df = pd.DataFrame({
        "ts_code": "02015.HK", "Date": dates,
        "Open": opens.round(2), "High": highs.round(2),
        "Low": lows.round(2), "Close": close_series.round(2),
        "Volume": volumes, "Amount": amounts, "Cash_Dividend": 0.0,
    })
    df["Date"] = pd.to_datetime(df["Date"])
    return df.reset_index(drop=True)


def run_one_scenario(df: pd.DataFrame, grid_up: float, grid_down: float,
                     add_pct: float, reduce_pct: float,
                     ts_code: str, stock_name: str, market: str,
                     pm_start_date: str, pm_initial_shares: int,
                     initial_capital: float, per_shares: int = 100) -> tuple:
    """独立回测函数，参数全部显式传入"""
    config = BacktestConfig(
        ts_code=ts_code, stock_name=stock_name, market=market,
        mode="position_management_1",
        pm_start_date=pm_start_date,
        pm_initial_shares=pm_initial_shares,
        initial_capital=initial_capital,
        per_shares=per_shares,
        data_source="akshare",
        pm_grid_up=grid_up, pm_grid_down=grid_down,
        pm_add_pct=add_pct, pm_reduce_pct=reduce_pct,
    )
    df = compute_indicators(
        df,
        entry_period=config.entry_period, exit_period=config.exit_period,
        atr_period=config.atr_period, ema_fast=config.ema_fast,
        ema_slow=config.ema_slow, channel_type=config.channel_type,
        bb_period=config.bb_period, bb_k=config.bb_k,
        am_ma_period=config.am_ma_period, mode=config.mode,
    )
    trades, equity = run_backtest(df, config)
    stats = calc_stats(trades, config.initial_capital, equity)
    return stats, trades, equity


# ==================== 分析主逻辑 ====================

def analyze_one_stock(ts_code: str, name: str, initial_shares: int,
                      initial_capital: float, per_shares: int,
                      grid_up: float, grid_down: float,
                      add_pct: float, reduce_pct: float):
    """对一个标的做完整分析"""
    from core.cache import load as cache_load

    df = cache_load(ts_code, "HK")
    if df is None or df.empty:
        print(f"❌ {name}: 无数据")
        return
    df = df.sort_values("Date").reset_index(drop=True)
    df = compute_indicators(df, mode="position_management_1")

    close = df["Close"].values
    dates = df["Date"].values
    current_price = float(close[-1])
    peak_1y = float(np.max(close[-252:]))

    adx_series = df["ADX14"].values
    plus_di = df["+DI"].values
    minus_di = df["-DI"].values

    current_adx = float(adx_series[-1]) if not np.isnan(adx_series[-1]) else 0
    current_pdi = float(plus_di[-1]) if not np.isnan(plus_di[-1]) else 0
    current_mdi = float(minus_di[-1]) if not np.isnan(minus_di[-1]) else 0

    # 找历史上 ADX>30 且 -DI > +DI 的下跌段
    bear_trend_mask = (
        (~np.isnan(adx_series)) & (adx_series > 30) &
        (~np.isnan(plus_di)) & (~np.isnan(minus_di)) &
        (minus_di > plus_di)
    )
    bear_periods = []
    in_bear, start_idx = False, 0
    for i in range(len(bear_trend_mask)):
        if bear_trend_mask[i] and not in_bear:
            start_idx = i; in_bear = True
        elif not bear_trend_mask[i] and in_bear:
            if i - start_idx >= 20:
                bear_periods.append((start_idx, i))
            in_bear = False
    if in_bear and len(bear_trend_mask) - start_idx >= 20:
        bear_periods.append((start_idx, len(bear_trend_mask)))

    print(f"\n{'='*70}")
    print(f"📊 {name} ({ts_code}) — 策略参数: "
          f"grid_up={grid_up} down={grid_down} add={add_pct} reduce={reduce_pct}")
    print(f"{'='*70}")
    print(f"  当前价: ¥{current_price:.2f}  |  1年最高: ¥{peak_1y:.2f}  "
          f"|  距高点: {(current_price/peak_1y-1)*100:.1f}%")
    print(f"  ADX={current_adx:.1f}  +DI={current_pdi:.1f}  -DI={current_mdi:.1f}  "
          f"→ {'强趋势下跌' if current_mdi > current_pdi else '趋势不明'}")
    print(f"  初始持仓: {initial_shares}手  |  可用资金: ¥{initial_capital:,}  |  "
          f"持仓市值: ¥{initial_shares * current_price * per_shares:,}")
    print(f"  历史类似下跌段: {len(bear_periods)} 段")

    # ---- 历史场景回测 ----
    print(f"\n  --- 历史回测：在类似下跌趋势中启动策略 vs 死扛 ---")
    scenarios = []
    for bs, be in bear_periods[-5:]:
        pm_start_idx = min(bs + 10, be - 10)
        price_at_start = float(close[pm_start_idx])
        peak_before = float(np.max(close[max(0, bs-60):bs]))
        dd_from_peak = (price_at_start / peak_before - 1) * 100
        if dd_from_peak > -15:
            continue

        end_idx = min(pm_start_idx + 126, len(df) - 1)
        pm_start_date = pd.Timestamp(dates[pm_start_idx]).strftime("%Y%m%d")
        config = BacktestConfig(
            ts_code=ts_code, stock_name=name, market="HK",
            mode="position_management_1", per_shares=per_shares,
            data_source="akshare",
            pm_grid_up=grid_up, pm_grid_down=grid_down,
            pm_add_pct=add_pct, pm_reduce_pct=reduce_pct,
            pm_start_date=pm_start_date,
            pm_initial_shares=initial_shares,
            initial_capital=initial_capital,
        )
        try:
            sub_df = df.iloc[:end_idx + 1].copy()
            trades, equity = run_backtest(sub_df, config)
            stats = calc_stats(trades, config.initial_capital, equity)
            end_price = float(close[end_idx])
            bh_return = (end_price / price_at_start - 1) * 100
            grid_return = stats["总收益率 (%)"]
            diff = grid_return - bh_return
            scenarios.append({
                "start_date": pd.Timestamp(dates[pm_start_idx]).strftime("%Y-%m-%d"),
                "start_price": price_at_start, "end_price": end_price,
                "dd_from_peak": dd_from_peak,
                "bh_return": bh_return, "grid_return": grid_return,
                "diff": diff, "trades": stats["总交易"],
                "max_dd": stats["最大回撤 (%)"],
            })
        except Exception as e:
            pass

    if scenarios:
        print(f"  {'启动日':<12} {'启动价':>8} {'终点价':>8} {'死扛':>9} {'网格':>9} {'差额':>9} {'交易':>5}")
        print(f"  {'-'*60}")
        wins, total_diff = 0, 0
        for s in scenarios:
            tag = "✓" if s["diff"] > 0 else "✗"
            print(f"  {s['start_date']:<12} {s['start_price']:>8.2f} {s['end_price']:>8.2f} "
                  f"{s['bh_return']:>+8.2f}% {s['grid_return']:>+8.2f}% "
                  f"{s['diff']:>+8.2f}% {tag} {s['trades']:>4}")
            if s["diff"] > 0: wins += 1
            total_diff += s["diff"]
        avg_diff = total_diff / len(scenarios)
        verdict = "✅ 策略有效" if avg_diff > 5 else "⚠️ 效果有限" if avg_diff > 0 else "❌ 策略有害"
        print(f"  {'─'*60}")
        print(f"  改善概率: {wins}/{len(scenarios)} ({wins/len(scenarios)*100:.0f}%)  "
              f"平均差额: {avg_diff:+.2f}%  {verdict}")
    else:
        print(f"  (无符合条件的下跌段)")

    # ---- 未来场景模拟（显式传参） ----
    print(f"\n  --- 未来3个月8种场景（从 ¥{current_price:.2f} 出发，参数同上） ---")
    SCENARIO_LABELS = [
        ("单边上涨", "up"), ("单边下跌", "down"),
        ("震荡向上", "volatile_up"), ("震荡向下", "volatile_down"),
        ("横盘震荡", "sideways"), ("缓慢阴跌", "slow_grind_down"),
        ("V形反转", "v_reversal"), ("倒V反转", "inv_v_reversal"),
    ]
    fwd_results = []
    for label, trend_type in SCENARIO_LABELS:
        sim_df = generate_ohlcv("20260624", 63, current_price, trend_type)
        stats, trades, equity = run_one_scenario(
            sim_df, grid_up=grid_up, grid_down=grid_down,
            add_pct=add_pct, reduce_pct=reduce_pct,
            ts_code=ts_code, stock_name=name, market="HK",
            pm_start_date="20260624", pm_initial_shares=initial_shares,
            initial_capital=initial_capital, per_shares=per_shares,
        )
        fwd_results.append((label, stats, sim_df))

    print(f"  {'形态':<10} {'价格变动':>10} {'策略收益':>10} {'盈亏':>12} {'最大回撤':>10} {'交易':>5} {'死扛':>10}")
    print(f"  {'-'*70}")
    for label, stats, sdf in fwd_results:
        price_chg = (sdf["Close"].iloc[-1] / sdf["Close"].iloc[0] - 1) * 100
        print(f"  {label:<10} {price_chg:>+9.2f}% {stats['总收益率 (%)']:>+9.2f}% "
              f"{stats['总盈亏 (¥)']:>+11,.0f} {stats['最大回撤 (%)']:>+9.2f}% "
              f"{stats['总交易']:>5} {price_chg:>+9.2f}%")
    profit_count = sum(1 for _, s, _ in fwd_results if s["总收益率 (%)"] > 0)
    print(f"  {'─'*70}")
    print(f"  8种场景盈利: {profit_count}/8  |  "
          f"{'✅ 多数盈利' if profit_count>=6 else '⚠️ 多数亏损' if profit_count<=2 else '➖ 喜忧参半'}")


# ==================== main ====================
if __name__ == "__main__":
    print("=" * 70)
    print("🔍 深套持仓 — 仓位管理策略适用性分析")
    print("   所有参数显式传入，不依赖 simulate_market_scenarios.py 的全局变量")
    print("=" * 70)

    # ─── 理想汽车：网格搜索最优参数 ───
    print("\n>>> 理想汽车 (grid_up=0.03 down=0.03 add=0.03 reduce=0.75)")
    analyze_one_stock(
        ts_code="02015.HK", name="理想汽车",
        initial_shares=26, initial_capital=40000, per_shares=100,
        grid_up=0.03, grid_down=0.03, add_pct=0.030, reduce_pct=0.75,
    )

    # ─── 老铺黄金：网格搜索最优参数 ───
    print("\n>>> 老铺黄金 (grid_up=0.02 down=0.05 add=0.03 reduce=0.75)")
    analyze_one_stock(
        ts_code="06181.HK", name="老铺黄金",
        initial_shares=3, initial_capital=40000, per_shares=100,
        grid_up=0.02, grid_down=0.05, add_pct=0.030, reduce_pct=0.75,
    )
