#!/usr/bin/env python3
"""配置函数（纯函数，无 IO，便于单测）

巴菲特1957信原则：市场低估 → 全仓低估类；市场走高 → 获利了结、转向套利类（MVP=现金）。
价值组合权重 w 是估值分位的分段线性递减函数。
"""
import pandas as pd


def expanding_percentile(values: pd.Series) -> pd.Series:
    """无前视分位：第 t 个值的分位 = 历史(含当期)中 ≤ 该值的占比，返回 [0,1]"""
    out = pd.Series(index=values.index, dtype=float)
    for t in range(len(values)):
        v = values.iloc[t]
        out.iloc[t] = (values.iloc[:t + 1] <= v).sum() / (t + 1)
    return out


def value_weight(pct: float, lo: float = 0.30, hi: float = 0.70,
                 w_min: float = 0.25) -> float:
    """估值分位 → 价值组合权重：
    pct ≤ lo → 1.0（低估全仓）；lo~hi 线性过渡；pct ≥ hi → w_min（高估时留最低仓位）"""
    if pct <= lo:
        return 1.0
    if pct >= hi:
        return w_min
    return w_min + (1.0 - w_min) * (hi - pct) / (hi - lo)


def monthly_weights(pe_series: pd.Series, lo: float = 0.30, hi: float = 0.70,
                    w_min: float = 0.25, min_history: int = 120) -> pd.Series:
    """月度权重序列：expanding 分位 → value_weight；
    历史不足 min_history 个月时按 0.5 分位处理（线性区间中点）。"""
    pct = expanding_percentile(pe_series)
    weights = pd.Series(index=pe_series.index, dtype=float)
    for t in range(len(pe_series)):
        p = pct.iloc[t] if t + 1 >= min_history else 0.5
        weights.iloc[t] = value_weight(p, lo, hi, w_min)
    return weights
