#!/usr/bin/env python3
"""自检：配置函数端点/单调性、分位数手工核对、缓存完整性、无前视口径"""
import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from core.cache import load as cache_load
from strategy.buffett_1957.allocation import value_weight, expanding_percentile, monthly_weights


def _check(name, cond):
    print(f"  {'✅' if cond else '❌'} {name}")
    if not cond:
        raise AssertionError(name)


def main():
    print("── 1. value_weight 端点与单调性 ──")
    _check("pct=0 → 1.0", value_weight(0.0) == 1.0)
    _check("pct=0.3 → 1.0", value_weight(0.3) == 1.0)
    _check("pct=0.5 → 线性中点 0.625", abs(value_weight(0.5) - 0.625) < 1e-9)
    _check("pct=0.7 → 0.25", value_weight(0.7) == 0.25)
    _check("pct=1.0 → 0.25", value_weight(1.0) == 0.25)
    ws = [value_weight(p) for p in np.linspace(0, 1, 21)]
    _check("单调非增", all(a >= b for a, b in zip(ws, ws[1:])))

    print("── 2. expanding_percentile 手工核对 ──")
    s = pd.Series([3.0, 2.0, 1.0])
    p = expanding_percentile(s)
    _check("[3,2,1] → [1, 0.5, 1/3]", abs(p.iloc[0] - 1.0) < 1e-9
           and abs(p.iloc[1] - 0.5) < 1e-9 and abs(p.iloc[2] - 1 / 3) < 1e-9)

    print("── 3. monthly_weights 无前视口径（只用 ≤t 数据） ──")
    pe = pd.Series([10.0, 9.0, 11.0, 12.0, 8.0], index=pd.date_range("2000-01-31", periods=5, freq="ME"))
    w = monthly_weights(pe, min_history=3)
    _check("前 2 个月历史不足 → 0.5 分位权重 0.625", all(abs(w.iloc[:2] - 0.625) < 1e-9))
    _check("第 3 个月（分位 1.0）→ 0.25", abs(w.iloc[2] - 0.25) < 1e-9)
    _check("第 5 个月（分位 0.2 ≤ lo）→ 1.0", abs(w.iloc[4] - 1.0) < 1e-9)

    print("── 4. 数据缓存完整性 ──")
    pe_df = cache_load("HSI_PE", "IDX")
    _check("HSI_PE 已缓存", pe_df is not None and not pe_df.empty)
    if pe_df is not None and not pe_df.empty:
        _check("HSI_PE 无 0 值", (pe_df["addLyrPe"] > 0).all())
        _check("HSI_PE 日期升序无重复", pe_df["Date"].is_monotonic_increasing
               and not pe_df["Date"].duplicated().any())
    hsi = cache_load("HSI", "IDX")
    _check("HSI 日线已缓存", hsi is not None and not hsi.empty)
    if hsi is not None and not hsi.empty:
        _check("HSI 日期升序无重复", hsi["Date"].is_monotonic_increasing
               and not hsi["Date"].duplicated().any())
    uni = cache_load("universe", "VAL")
    _check("universe 已缓存", uni is not None and not uni.empty)

    print("\n✅ 自检全部通过")


if __name__ == "__main__":
    main()
