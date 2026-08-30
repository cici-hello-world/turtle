#!/usr/bin/env python3
"""筛选版巴菲特策略：流动性≥1亿 + 市值≥1000亿 → 估值最优 2 只 + 估值分位调仓

用法：
    PYTHONPATH=. python strategy/buffett_1957/run_filtered.py
输出：result/buffett_1957/filtered_*.csv（分年收益、季度持仓、净值）
"""
import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from core.statistics import calc_stats
from strategy.buffett_1957.config import Buffett1957Config
from strategy.buffett_1957.hsi_pe import load_hsi_pe, load_hsi_price, get_monthly_weights
from strategy.buffett_1957.hk_valuation import build_filtered_selection
from strategy.buffett_1957.backtest import run_backtest

MIN_TURNOVER = 1e8     # 日均成交额 ≥ 1 亿港元
MIN_MCAP = 1e11        # 总市值 ≥ 1000 亿港元（元）
OUT_DIR = os.path.join(PROJECT_ROOT, "result", "buffett_1957")


def main():
    cfg = Buffett1957Config(start_date="20200101", basket_n=2,
                            initial_capital=10_000.0)
    print("=" * 72)
    print(f"筛选版巴菲特策略  2020-01-01 → {cfg.end_date}")
    print(f"过滤: 日均成交额≥{MIN_TURNOVER/1e8:.0f}亿 且 总市值≥{MIN_MCAP/1e11:.0f}千亿 → "
          f"每季估值最优 {cfg.basket_n} 只")
    print("=" * 72)

    pe_df, src = load_hsi_pe()
    weights, pct, _ = get_monthly_weights(cfg, pe_df, src)
    selections = build_filtered_selection(cfg, MIN_TURNOVER, MIN_MCAP)

    trades, eq, eq_dates = run_backtest(cfg, weights, selections=selections)

    hsi = load_hsi_price()
    hsi = hsi[(hsi["Date"] >= pd.to_datetime(cfg.start_date))
              & (hsi["Date"] <= pd.to_datetime(cfg.end_date))]
    bench = (hsi["Close"].iloc[-1] / hsi["Close"].iloc[0] - 1) * 100
    stats = calc_stats(trades, cfg.initial_capital, eq, eq_dates, benchmark_return=bench)

    print(f"\n整体: 收益 {stats['总收益率 (%)']:.1f}%  MaxDD {stats['最大回撤 (%)']:.1f}%  "
          f"稳健夏普 {stats['稳健夏普比率']:.2f}  交易 {stats['总交易']}  "
          f"期末 {stats['最终资金 (¥)']:,.0f}  (恒指 {bench:.1f}%)")

    # 分年收益（按主日历自然年）
    eq_s = pd.Series(eq, index=pd.to_datetime(eq_dates))
    hsi_s = hsi.set_index("Date")["Close"]
    hsi_s = hsi_s[hsi_s.index >= eq_s.index.min()]
    rows = []
    for y, sub in eq_s.groupby(eq_s.index.year):
        if len(sub) < 2:
            continue
        ret = (sub.iloc[-1] / sub.iloc[0] - 1) * 100
        hm = hsi_s[hsi_s.index.year == y]
        hret = (hm.iloc[-1] / hm.iloc[0] - 1) * 100 if len(hm) >= 2 else float("nan")
        rows.append({"年份": y, "策略收益%": round(ret, 1),
                     "恒指%": round(hret, 1), "超额%": round(ret - hret, 1)})
    yr = pd.DataFrame(rows)
    print("\n分年收益:")
    print(yr.to_string(index=False))

    # 季度持仓与成交额/市值
    hold_rows = []
    for qe, sel in selections.items():
        for _, r in sel.iterrows():
            hold_rows.append({"quarter": qe, "ts_code": r["ts_code"],
                              "pe_ttm": r["pe_ttm"], "pb": r["pb"]})
    pd.DataFrame(hold_rows).to_csv(os.path.join(OUT_DIR, "filtered_holdings.csv"), index=False)
    yr.to_csv(os.path.join(OUT_DIR, "filtered_yearly.csv"), index=False)
    eq_s.to_frame("equity").to_csv(os.path.join(OUT_DIR, "filtered_equity.csv"))
    print(f"\n✅ 输出: {OUT_DIR}/filtered_*.csv")


if __name__ == "__main__":
    main()
