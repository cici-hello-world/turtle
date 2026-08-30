#!/usr/bin/env python3
"""报告：三条净值曲线对比 + 月度权重表 + 持仓明细 + 统计汇总

输出 result/buffett_1957/：
  nav_curves.csv / nav_curves.png
  monthly_weights.csv
  quarterly_holdings.csv
  summary.csv
"""
import os

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(PROJECT_ROOT, "result", "buffett_1957")


def _norm(equity, dates) -> pd.Series:
    return pd.Series(equity, index=dates) / equity[0] * 100


def make_report(config, trades, eq, eq_dates, value_eq, value_dates, hsi_df,
                weights: pd.Series, selections: dict) -> dict:
    os.makedirs(OUT_DIR, exist_ok=True)

    hsi = hsi_df.set_index("Date")["Close"]
    hsi = hsi[(hsi.index >= pd.to_datetime(config.start_date))
              & (hsi.index <= pd.to_datetime(config.end_date))]
    hsi_nav = hsi / hsi.iloc[0] * 100

    nav = pd.DataFrame({
        "巴菲特组合": _norm(eq, eq_dates),
        "100%价值组合": _norm(value_eq, value_dates),
        "恒指买入持有": hsi_nav,
    }).sort_index()
    nav.to_csv(os.path.join(OUT_DIR, "nav_curves.csv"))

    weights.to_frame("value_weight").to_csv(os.path.join(OUT_DIR, "monthly_weights.csv"))

    rows = []
    for qe, sel in selections.items():
        for _, r in sel.iterrows():
            rows.append({"quarter": qe, "ts_code": r["ts_code"],
                         "pe_ttm": r["pe_ttm"], "pb": r["pb"]})
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "quarterly_holdings.csv"), index=False)

    # 净值图
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(14, 7))
    for col in nav.columns:
        ax.plot(nav.index, nav[col], label=col, linewidth=1.2)
    ax.set_title("巴菲特1957策略 vs 基准（净值=100 起点）")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "nav_curves.png"), dpi=120)
    print(f"✅ 报告已保存: {OUT_DIR}")
    return {"nav": nav}
