#!/usr/bin/env python3
"""网格策略适用性验证脚本

验证维度：
  1. 按市场状态（趋势/震荡）拆分的超额收益
  2. 参数稳健性（不同网格宽度）
  3. 最差情况分析（最大连续亏损、最长水下期）
  4. 资金效率（现金闲置率）

用法：
    python validate_grid_strategy.py
"""

import pandas as pd
import numpy as np
from datetime import timedelta

from core.config import BacktestConfig
from core.data import fetch_data
from core.indicators import compute_indicators
from core.engine import run_backtest


def fetch_all_data(ts_code: str, stock_name: str, market: str,
                   roll_start: str, window_months: int = 3) -> pd.DataFrame:
    """拉取全部数据"""
    roll_start_dt = pd.to_datetime(roll_start)
    fetch_start = roll_start_dt.strftime("%Y%m%d")
    fetch_end = (pd.Timestamp.today()- pd.DateOffset(days=1)).strftime("%Y%m%d")
    exchange = "XHKG" if market == "HK" else "XSHG"
    print(f"📥 获取 {stock_name}({ts_code}) {fetch_start} ~ {fetch_end} ...")
    df = fetch_data(ts_code=ts_code, stock_name=stock_name, market=market,
                    start_date=fetch_start, end_date=fetch_end, exchange=exchange,data_source="akshare")
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").reset_index(drop=True)


def run_single_window(df_window: pd.DataFrame, config: BacktestConfig):
    """运行单个窗口，返回结果字典"""
    df_w = compute_indicators(df_window.copy(), channel_type="donchian",
                               mode="position_management_1")
    trades, equity = run_backtest(df_w, config)
    closes = df_window["Close"].values
    price_change = (closes[-1] - closes[0]) / closes[0] if len(closes) > 1 else 0.0
    if equity:
        total_return = (equity[-1] - equity[0]) / equity[0] if len(trades) > 0 else price_change
    else:
        total_return = 0.0

    # 窗口期平均 ADX（判断趋势强度）
    adx_vals = df_w["ADX14"].dropna().values if "ADX14" in df_w.columns else np.array([])
    avg_adx = float(np.mean(adx_vals)) if len(adx_vals) > 0 else 0.0

    # 窗口期股价路径：线性回归斜率判断趋势方向
    t = np.arange(len(closes))
    if len(t) > 2:
        slope = np.polyfit(t, closes, 1)[0]
        trend_dir = "up" if slope > 0 else "down"
    else:
        trend_dir = "flat"

    return {
        "total_return": total_return,
        "price_change": price_change,
        "excess_return": total_return - price_change,
        "avg_adx": avg_adx,
        "trend_dir": trend_dir,
        "trades": len(trades),
    }


def validate_regime(df: pd.DataFrame, ts_code: str, stock_name: str, market: str,
                    roll_start: str, window_months: int = 3, step_days: int = 5):
    """维度1：按市场状态拆分超额收益"""
    print("\n" + "=" * 60)
    print("📊 维度1：按市场状态拆分收益")
    print("=" * 60)

    roll_start_dt = pd.to_datetime(roll_start)
    last_data_date = df["Date"].max()
    last_window_start = last_data_date - pd.DateOffset(months=window_months)

    window_starts = []
    current = roll_start_dt
    while current <= last_window_start:
        window_starts.append(current)
        current += timedelta(days=step_days)

    results = []
    for ws in window_starts:
        we = ws + pd.DateOffset(months=window_months)
        mask = (df["Date"] >= ws) & (df["Date"] <= we)
        df_window = df[mask].copy()
        if len(df_window) < 20:
            continue
        config = BacktestConfig(
            mode="position_management_1", ts_code=ts_code, stock_name=stock_name,
            market=market, pm_start_date=ws.strftime("%Y%m%d"),
            pm_initial_shares=3 if market == "HK" else 9, initial_capital=50000,
            pm_use_dmi=False,
        )
        r = run_single_window(df_window, config)
        r["start_date"] = ws.strftime("%Y-%m-%d")
        results.append(r)

    df_r = pd.DataFrame(results)

    # 按 ADX 分组：ADX<20 震荡，20-30 弱趋势，>30 强趋势
    df_r["regime"] = pd.cut(df_r["avg_adx"], bins=[0, 20, 30, 100],
                             labels=["震荡(ADX<20)", "弱趋势(20-30)", "强趋势(>30)"])
    print("\n按趋势强度分组：")
    for regime in ["震荡(ADX<20)", "弱趋势(20-30)", "强趋势(>30)"]:
        sub = df_r[df_r["regime"] == regime]
        if len(sub) == 0:
            continue
        print(f"\n  {regime} ({len(sub)}个窗口):")
        print(f"    平均超额: {sub['excess_return'].mean():.2%}  |  "
              f"中位数超额: {sub['excess_return'].median():.2%}")
        print(f"    胜率: {(sub['excess_return'] > 0).mean():.1%}  |  "
              f"最大超额: {sub['excess_return'].max():.2%}  |  "
              f"最小超额: {sub['excess_return'].min():.2%}")

    # 按趋势方向分组
    print("\n按趋势方向分组：")
    for direction in ["up", "down", "flat"]:
        sub = df_r[df_r["trend_dir"] == direction]
        if len(sub) == 0:
            continue
        print(f"\n  {direction}趋势 ({len(sub)}个窗口):")
        print(f"    平均超额: {sub['excess_return'].mean():.2%}  |  "
              f"中位数超额: {sub['excess_return'].median():.2%}")
        print(f"    跑赢持股: {(sub['excess_return'] > 0).mean():.1%}")

    # 组合分组（最重要的判断依据）
    print("\n🔑 组合分组（趋势 × 强度）：")
    for regime in ["震荡(ADX<20)", "弱趋势(20-30)", "强趋势(>30)"]:
        for direction in ["up", "down"]:
            sub = df_r[(df_r["regime"] == regime) & (df_r["trend_dir"] == direction)]
            if len(sub) < 3:
                continue
            print(f"  {regime} + {direction}: "
                  f"平均超额{sub['excess_return'].mean():.2%}, "
                  f"胜率{(sub['excess_return'] > 0).mean():.0%} "
                  f"({len(sub)}窗)")


def validate_params(df: pd.DataFrame, ts_code: str, stock_name: str, market: str,
                    roll_start: str, window_months: int = 3):
    """维度2：参数稳健性——不同网格宽度"""
    print("\n" + "=" * 60)
    print("📊 维度2：参数稳健性（不同网格宽度）")
    print("=" * 60)

    roll_start_dt = pd.to_datetime(roll_start)
    last_data_date = df["Date"].max()
    last_window_start = last_data_date - pd.DateOffset(months=window_months)

    window_starts = []
    current = roll_start_dt
    while current <= last_window_start:
        window_starts.append(current)
        current += timedelta(days=10)  # 每10天采样，减少计算量

    grid_configs = [
        (0.03, 0.03, "窄网格 ±3%"),
        (0.05, 0.05, "标准 ±5%"),
        (0.08, 0.08, "宽网格 ±8%"),
        (0.03, 0.06, "非对称 +3%/-6%"),
    ]

    for grid_up, grid_down, label in grid_configs:
        excesses = []
        for ws in window_starts:
            we = ws + pd.DateOffset(months=window_months)
            mask = (df["Date"] >= ws) & (df["Date"] <= we)
            df_window = df[mask].copy()
            if len(df_window) < 20:
                continue
            config = BacktestConfig(
                mode="position_management_1", ts_code=ts_code, stock_name=stock_name,
                market=market, pm_start_date=ws.strftime("%Y%m%d"),
                pm_initial_shares=3 if market == "HK" else 9, initial_capital=50000,
                pm_grid_up=grid_up, pm_grid_down=grid_down, pm_use_dmi=False,
            )
            r = run_single_window(df_window, config)
            excesses.append(r["excess_return"])

        print(f"  {label}: 平均超额={np.mean(excesses):.2%}, "
              f"中位数={np.median(excesses):.2%}, "
              f"胜率={(np.array(excesses)>0).mean():.1%}")


def validate_worst_case(df: pd.DataFrame, ts_code: str, stock_name: str, market: str,
                        roll_start: str, window_months: int = 3, step_days: int = 5):
    """维度3：最差情况分析"""
    print("\n" + "=" * 60)
    print("📊 维度3：最差情况分析")
    print("=" * 60)

    roll_start_dt = pd.to_datetime(roll_start)
    last_data_date = df["Date"].max()
    last_window_start = last_data_date - pd.DateOffset(months=window_months)

    window_starts = []
    current = roll_start_dt
    while current <= last_window_start:
        window_starts.append(current)
        current += timedelta(days=step_days)

    excesses = []
    drawdowns = []
    for ws in window_starts:
        we = ws + pd.DateOffset(months=window_months)
        mask = (df["Date"] >= ws) & (df["Date"] <= we)
        df_window = df[mask].copy()
        if len(df_window) < 20:
            continue
        config = BacktestConfig(
            mode="position_management_1", ts_code=ts_code, stock_name=stock_name,
            market=market, pm_start_date=ws.strftime("%Y%m%d"),
            pm_initial_shares=3 if market == "HK" else 9, initial_capital=50000,
            pm_use_dmi=False,
        )
        df_w = compute_indicators(df_window.copy(), channel_type="donchian",
                                   mode="position_management_1")
        trades, equity = run_backtest(df_w, config)
        if len(equity) < 2:
            continue
        # 窗口内最大回撤
        eq = pd.Series(equity)
        dd = (eq - eq.cummax()) / eq.cummax()
        drawdowns.append(dd.min())
        closes = df_window["Close"].values
        pc = (closes[-1] - closes[0]) / closes[0] if len(closes) > 1 else 0
        tr = (equity[-1] - equity[0]) / equity[0] if len(trades) > 0 else pc
        excesses.append(tr - pc)

    excesses = np.array(excesses)
    drawdowns = np.array(drawdowns)

    print(f"  最差5个窗口超额收益: {np.sort(excesses)[:5]}")
    print(f"  超额收益 VaR(5%): {np.percentile(excesses, 5):.2%}")
    print(f"  超额收益 VaR(1%): {np.percentile(excesses, 1):.2%}")
    print(f"  平均最大回撤: {np.mean(drawdowns):.2%}")
    print(f"  极端最大回撤: {np.min(drawdowns):.2%}")
    print(f"  超额正偏度: {np.mean(excesses > 0):.1%} 窗口的超额>0")
    # 连续负超额窗口
    neg_streak = 0
    max_neg_streak = 0
    for e in excesses:
        if e < 0:
            neg_streak += 1
            max_neg_streak = max(max_neg_streak, neg_streak)
        else:
            neg_streak = 0
    print(f"  最长连续负超额窗口: {max_neg_streak}")


def validate_cash_efficiency(df: pd.DataFrame, ts_code: str, stock_name: str, market: str,
                              roll_start: str, window_months: int = 3, step_days: int = 10):
    """维度4：资金效率"""
    print("\n" + "=" * 60)
    print("📊 维度4：资金效率")
    print("=" * 60)

    roll_start_dt = pd.to_datetime(roll_start)
    last_data_date = df["Date"].max()
    last_window_start = last_data_date - pd.DateOffset(months=window_months)

    window_starts = []
    current = roll_start_dt
    while current <= last_window_start:
        window_starts.append(current)
        current += timedelta(days=step_days)

    cash_ratios = []
    trade_counts = []
    for ws in window_starts:
        we = ws + pd.DateOffset(months=window_months)
        mask = (df["Date"] >= ws) & (df["Date"] <= we)
        df_window = df[mask].copy()
        if len(df_window) < 20:
            continue
        config = BacktestConfig(
            mode="position_management_1", ts_code=ts_code, stock_name=stock_name,
            market=market, pm_start_date=ws.strftime("%Y%m%d"),
            pm_initial_shares=3 if market == "HK" else 9, initial_capital=50000,
            pm_use_dmi=False,
        )
        df_w = compute_indicators(df_window.copy(), channel_type="donchian",
                                   mode="position_management_1")
        trades, equity = run_backtest(df_w, config)
        if trades:
            # 平均现金/权益比
            cash_ratios.append(np.mean([t["Capital_After"] / t["Equity_After"]
                                        for t in trades if t["Equity_After"] > 0]))
        trade_counts.append(len(trades))

    if cash_ratios:
        print(f"  平均现金/权益比: {np.mean(cash_ratios):.1%}")
        print(f"  最低现金/权益比: {np.min(cash_ratios):.1%}")
    print(f"  平均每窗交易次数: {np.mean(trade_counts):.1f}")
    print(f"  零交易窗口占比: {(np.array(trade_counts)==0).mean():.1%}")


# ============================================================
if __name__ == "__main__":
    # 可修改要验证的标的
    TS_CODE = "06181.HK"
    STOCK_NAME = "老铺"
    MARKET = "HK"
    ROLL_START = "20240628"
    WINDOW_MONTHS = 6

    df_all = fetch_all_data(TS_CODE, STOCK_NAME, MARKET, ROLL_START, WINDOW_MONTHS)

    validate_regime(df_all, TS_CODE, STOCK_NAME, MARKET, ROLL_START, WINDOW_MONTHS)
    validate_params(df_all, TS_CODE, STOCK_NAME, MARKET, ROLL_START, WINDOW_MONTHS)
    validate_worst_case(df_all, TS_CODE, STOCK_NAME, MARKET, ROLL_START, WINDOW_MONTHS)
    validate_cash_efficiency(df_all, TS_CODE, STOCK_NAME, MARKET, ROLL_START, WINDOW_MONTHS)

    print("\n" + "=" * 60)
    print("🎯 采用网格策略的判断标准：")
    print("=" * 60)
    print("  ✅ 震荡市超额 > 5%，且胜率 > 60%")
    print("  ✅ 下跌趋势中超额 > 0%（能减少亏损）")
    print("  ✅ 强趋势上涨市中超额可以负（网格跑输持有），这是预期内的")
    print("  ✅ 参数稳健：不同网格宽度超额方向一致")
    print("  ✅ 最差5%窗口超额可控（非灾难性亏损）")
    print("  ✅ 资金利用合理（不会长期满仓也不会长期空仓）")
    print("  ❌ 如果在震荡市也跑不赢 → 标的波动不够，不适合网格")
    print("  ❌ 如果下跌趋势中超额 < -10% → 网格加剧亏损，不适合")
