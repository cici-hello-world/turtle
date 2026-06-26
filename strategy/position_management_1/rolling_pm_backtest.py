#!/usr/bin/env python3
"""仓位管理策略1 滚动回测

从指定起始日期开始，以固定窗口滚动运行 position_management_1 策略，
返回每个窗口的总收益率数组。

用法：
    python rolling_pm_backtest.py
"""

import pandas as pd
import numpy as np
from datetime import timedelta

from core.config import BacktestConfig
from core.data import fetch_data
from core.indicators import compute_indicators
from core.engine import run_backtest


def rolling_backtest(
    ts_code: str = "02015.HK",
    stock_name: str = "理想",
    market: str = "HK",
    roll_start: str = "20210812",
    window_months: int = 9,
    step_days: int = 30,
    pm_initial_shares: int = 26,
    initial_capital: float = 100000,
    pm_use_dmi=False,
) -> list[dict]:
    """滚动回测仓位管理策略1。

    返回列表，每个元素为 {"start_date": str, "end_date": str, "total_return": float}
    """
    roll_start_dt = pd.to_datetime(roll_start)

    # 一次性拉取全部所需数据（从 roll_start 前 1 个月开始，到最新）
    fetch_start = roll_start_dt.strftime("%Y%m%d")
    fetch_end = (pd.Timestamp.today()-pd.DateOffset(days=1)).strftime("%Y%m%d")

    print(f"📥 获取 {stock_name}({ts_code}) 数据 {fetch_start} ~ {fetch_end} ...")
    exchange = "XHKG" if market == "HK" else "XSHG"
    df_all = fetch_data(
        ts_code=ts_code,
        stock_name=stock_name,
        market=market,
        start_date=fetch_start,
        end_date=fetch_end,
        exchange=exchange,
    )

    # 日期列转为 Timestamp 以便切片
    df_all["Date"] = pd.to_datetime(df_all["Date"])
    df_all = df_all.sort_values("Date").reset_index(drop=True)

    # 确定最后一个窗口的起始日
    last_data_date = df_all["Date"].max()
    last_window_start = last_data_date - pd.DateOffset(months=window_months)

    # 生成所有窗口起始日期
    window_starts = []
    current = roll_start_dt
    while current <= last_window_start:
        window_starts.append(current)
        current += timedelta(days=step_days)

    print(f"🔁 共 {len(window_starts)} 个窗口，窗口={window_months}个月，步长={step_days}天")

    results = []
    ps = 100  # HK 每手股数

    for i, ws in enumerate(window_starts):
        we = ws + pd.DateOffset(months=window_months)

        # 切片数据
        mask = (df_all["Date"] >= ws) & (df_all["Date"] <= we)
        df_window = df_all[mask].copy()

        if len(df_window) < 10:
            continue

        # 计算指标
        df_window = compute_indicators(df_window, channel_type="donchian")

        # 运行策略
        config = BacktestConfig(
            mode="position_management_1",
            ts_code=ts_code,
            stock_name=stock_name,
            market=market,
            pm_start_date=ws.strftime("%Y%m%d"),
            pm_initial_shares=pm_initial_shares,
            initial_capital=initial_capital,
            pm_use_dmi=pm_use_dmi,
        )
        trades, equity = run_backtest(df_window, config)

        # 总收益率 = (最终权益 - 初始权益) / 初始权益
        if equity:
            total_return = (equity[-1] - equity[0]) / equity[0]
        else:
            total_return = 0.0

        # 窗口期股价变动率
        closes = df_window["Close"].values
        price_change = (closes[-1] - closes[0]) / closes[0] if len(closes) > 1 else 0.0

        results.append({
            "start_date": ws.strftime("%Y-%m-%d"),
            "end_date": we.strftime("%Y-%m-%d"),
            "total_return": round(total_return, 6),
            "price_change": round(price_change, 6),
            "excess_return": round(total_return - price_change, 6),
            "trades": len(trades),
        })

        if (i + 1) % 50 == 0:
            print(f"  进度: {i + 1}/{len(window_starts)}")

    # 汇总
    returns_arr = [r["total_return"] for r in results]
    price_arr = [r["price_change"] for r in results]
    excess_arr = [r["excess_return"] for r in results]
    print(f"\n📊 滚动回测完成：{len(results)} 个窗口")
    print(f"   平均收益率: {np.mean(returns_arr):.2%}  |  股价变动: {np.mean(price_arr):.2%}  |  超额: {np.mean(excess_arr):.2%}")
    print(f"   中位数收益率: {np.median(returns_arr):.2%}  |  股价变动: {np.median(price_arr):.2%}  |  超额: {np.median(excess_arr):.2%}")
    print(f"   最大收益率: {np.max(returns_arr):.2%}  |  超额最大: {np.max(excess_arr):.2%}")
    print(f"   最小收益率: {np.min(returns_arr):.2%}  |  超额最小: {np.min(excess_arr):.2%}")
    print(f"   胜率(>0): {sum(1 for r in returns_arr if r > 0) / len(returns_arr):.1%}")
    print(f"   跑赢持股: {sum(1 for e in excess_arr if e > 0) / len(excess_arr):.1%}")

    return results


if __name__ == "__main__":
    results = rolling_backtest()

    # 保存结果
    df_out = pd.DataFrame(results)
    out_path = "position_management_result/rolling_pm_returns.csv"
    import os
    os.makedirs("position_management_result", exist_ok=True)
    df_out.to_csv(out_path, index=False)
    print(f"\n💾 结果已保存至 {out_path}")
