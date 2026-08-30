#!/usr/bin/env python3
"""巴菲特1957组合策略回测主脚本

用法：
    PYTHONPATH=. python strategy/buffett_1957/run_backtest.py                  # 默认 2005→今
    PYTHONPATH=. python strategy/buffett_1957/run_backtest.py --start 20160101
    PYTHONPATH=. python strategy/buffett_1957/run_backtest.py --force          # 强制重拉数据
"""
import argparse
import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from core.statistics import calc_stats
from strategy.buffett_1957.config import Buffett1957Config
from strategy.buffett_1957.hsi_pe import load_hsi_pe, load_hsi_price, get_monthly_weights
from strategy.buffett_1957.hk_valuation import build_quarterly_selection
from strategy.buffett_1957.backtest import run_backtest
from strategy.buffett_1957.report import make_report


def main():
    parser = argparse.ArgumentParser(description="巴菲特1957组合策略回测")
    parser.add_argument("--start", type=str, default="20050101", help="回测起点 YYYYMMDD")
    parser.add_argument("--end", type=str, default="20260828", help="回测终点 YYYYMMDD")
    parser.add_argument("--universe-n", type=int, default=400, help="季度流动性股票池 Top-N")
    parser.add_argument("--basket-n", type=int, default=30, help="价值组合持仓数")
    parser.add_argument("--force", action="store_true", help="强制重拉数据（忽略缓存）")
    args = parser.parse_args()

    cfg = Buffett1957Config(start_date=args.start, end_date=args.end,
                            universe_n=args.universe_n, basket_n=args.basket_n)

    print("=" * 72)
    print(f"巴菲特1957组合策略回测  {cfg.start_date} → {cfg.end_date}")
    print("=" * 72)

    # 1. 估值权重
    pe_df, src = load_hsi_pe()
    weights, pct, src = get_monthly_weights(cfg, pe_df, src)
    print(f"估值来源: {src}；当前价值权重={weights.iloc[-1]:.2f}（分位 {pct.iloc[-1]:.2f}）")

    # 2. 季度选股（含股票池/估值数据，--force 时重拉）
    selections = build_quarterly_selection(cfg, force=args.force)
    print(f"季度选股: {len(selections)} 个季度")

    # 3. 巴菲特组合回测
    trades, eq, eq_dates = run_backtest(cfg, weights, selections=selections)

    # 4. 100% 价值组合对照（同篮子，权重恒 1）
    trades_v, eq_v, eq_dates_v = run_backtest(cfg, value_only=True, selections=selections)

    # 5. 基准与统计
    hsi = load_hsi_price()
    hsi = hsi[(hsi["Date"] >= pd.to_datetime(cfg.start_date))
              & (hsi["Date"] <= pd.to_datetime(cfg.end_date))]
    bench = (hsi["Close"].iloc[-1] / hsi["Close"].iloc[0] - 1) * 100

    st_b = calc_stats(trades, cfg.initial_capital, eq, eq_dates, benchmark_return=bench)
    st_v = calc_stats(trades_v, cfg.initial_capital, eq_v, eq_dates_v, benchmark_return=bench)

    print("\n" + "=" * 72)
    print("回测结果对比")
    print("=" * 72)
    print(f"{'策略':<14s}{'收益':>10s}{'MaxDD':>10s}{'稳健夏普':>10s}{'CAGR':>8s}{'交易':>7s}{'期末':>14s}")
    for label, st in [("巴菲特组合", st_b), ("100%价值组合", st_v)]:
        print(f"{label:<14s}{st['总收益率 (%)']:>9.1f}%{st['最大回撤 (%)']:>9.1f}%"
              f"{st['稳健夏普比率']:>10.2f}{st['CAGR (%)']:>7.1f}%{st['总交易']:>7d}"
              f"{st['最终资金 (¥)']:>14,.0f}")
    print(f"{'恒指买入持有':<14s}{bench:>9.1f}%")
    print(f"超额(巴菲特-恒指) = {st_b['超额收益 (%)']:+.1f}%")

    # 6. 报告
    make_report(cfg, trades, eq, eq_dates, eq_v, eq_dates_v, hsi, weights, selections)


if __name__ == "__main__":
    main()
