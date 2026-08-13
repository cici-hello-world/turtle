#!/usr/bin/env python3
"""pm_bb_* 参数诊断 —— 只检查「当前参数是否合适」，不做寻优。

用法：
    python strategy/position_management_1/check_bb_params.py --ts-code 03690.HK
    python strategy/position_management_1/check_bb_params.py --ts-code 09992.HK

数据来源：data_cache 缓存中的历史日线（不联网，缓存缺失会直接报错）。

检查内容（按 core/indicators.py 中 BB_Width_Pct 的同一公式复现）：
  1. 网格乘数 bb_mult 的分布是否撑满 [pm_bb_width_min, pm_bb_width_max] 映射区间
  2. 归一化是否经常饱和（贴 0/贴 1）——饱和说明映射区间或回看窗口不合适
  3. 乘数日间跳动幅度——跳动过大说明网格一天一变，交易信号不稳定
  4. 当前波动率状态（网格正在被放大还是收紧）
  5. period / k / lookback 参数敏感性对比——当前组合与替代组合的差异
  6. 分年波动率结构——判断 lookback 是否与波动率结构匹配

结论只是诊断（该不该调、往哪个方向调），真要找最优参数用 tune_bb_params.py。
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from core.cache import load as cache_load
from core.config import BacktestConfig
from strategy.position_management_1 import common_config

# ── 敏感性对比用的候选值（会自动并入当前配置值） ──
ALT_PERIODS = [5, 10, 20]
ALT_KS = [1.5, 2.0, 2.5]
ALT_LOOKBACKS = [21, 63, 126, 252]


def _bb_width(close: pd.Series, period: int, k: float) -> pd.Series:
    """BB 宽度 = (上轨-下轨)/中轨，与 indicators.py 一致。"""
    mid = close.rolling(window=period, min_periods=period).mean()
    std = close.rolling(window=period, min_periods=period).std()
    return ((mid + k * std) - (mid - k * std)) / mid


def _width_pct(width: pd.Series, lookback: int) -> pd.Series:
    """宽度 z-score 归一化 → [0,1]，与 indicators.py 一致。"""
    ma = width.rolling(window=lookback, min_periods=20).mean()
    sd = width.rolling(window=lookback, min_periods=20).std().replace(0, np.nan)
    norm = ((width - ma) / sd).clip(-2, 2)
    return (norm + 2) / 4


def _mult(pct: pd.Series, wmin: float, wmax: float) -> pd.Series:
    return wmin + pct * (wmax - wmin)


def _diagnose(cfg, df: pd.DataFrame):
    """输出全部诊断信息，返回当前乘数序列（供敏感性对比用）。"""
    close = df["Close"]
    cur = dict(
        period=cfg.pm_bb_period, k=cfg.pm_bb_k, lookback=cfg.pm_bb_width_lookback,
        wmin=cfg.pm_bb_width_min, wmax=cfg.pm_bb_width_max,
    )
    print(f"当前配置: pm_bb_period={cur['period']}, pm_bb_k={cur['k']}, "
          f"pm_bb_width_lookback={cur['lookback']}, "
          f"pm_bb_width_min={cur['wmin']}, pm_bb_width_max={cur['wmax']}")

    w_cur = _bb_width(close, cur["period"], cur["k"])
    p_cur = _width_pct(w_cur, cur["lookback"])
    m_cur = _mult(p_cur, cur["wmin"], cur["wmax"])
    valid = m_cur.dropna()
    if valid.empty:
        print("  ⚠️ 有效数据不足，无法诊断（历史太短？）")
        return m_cur

    # ── 1. 乘数分布与饱和 ──
    print(f"\n── 网格乘数 bb_mult 分布 (n={len(valid)}) ──")
    print(f"  mean={valid.mean():.3f} median={valid.median():.3f} "
          f"min={valid.min():.3f} max={valid.max():.3f}")
    print(f"  p10={valid.quantile(.1):.3f} p25={valid.quantile(.25):.3f} "
          f"p75={valid.quantile(.75):.3f} p90={valid.quantile(.9):.3f}")
    sat_lo = (p_cur <= 0.01).sum()
    sat_hi = (p_cur >= 0.99).sum()
    sat_ratio = (sat_lo + sat_hi) / valid.count()
    print(f"  BB_Width_Pct 饱和天数: 贴0(≤0.01)={sat_lo}, 贴1(≥0.99)={sat_hi} "
          f"({100*sat_ratio:.1f}%)")

    # ── 2. 日间跳动 ──
    dmult = m_cur.diff().abs().dropna()
    print(f"  |Δmult| 日间: mean={dmult.mean():.4f} p95={dmult.quantile(.95):.4f} "
          f"max={dmult.max():.4f}")
    print(f"  单日变动>0.1: {(dmult > 0.1).sum()} 天 ({100*(dmult > 0.1).mean():.1f}%)")

    # ── 3. 当前状态（最近 30 日 + 对实际网格的影响） ──
    print(f"\n── 最近 30 个交易日 ──")
    tail = df.assign(width=w_cur, pct=p_cur, mult=m_cur).tail(30)
    out = tail[["Date", "Close", "width", "pct", "mult"]].copy()
    out["Close"] = out["Close"].round(2)
    out[["width", "pct", "mult"]] = out[["width", "pct", "mult"]].round(3)
    print(out.to_string(index=False))
    print(f"\n  当前: BB宽度={w_cur.iloc[-1]:.4f} Pct={p_cur.iloc[-1]:.3f} "
          f"mult={m_cur.iloc[-1]:.3f} (网格×{m_cur.iloc[-1]:.2f})")
    gd = cfg.pm_grid_down
    gu = cfg.pm_grid_up
    m_now = m_cur.iloc[-1]
    print(f"  实际网格: 当前 down={gd*m_now:.3f} up={gu*m_now:.3f} "
          f"(区间 down={gd*cur['wmin']:.3f}~{gd*cur['wmax']:.3f}, "
          f"up={gu*cur['wmin']:.3f}~{gu*cur['wmax']:.3f})")

    # ── 4. 宽度本身分布 ──
    print(f"\n── BB 宽度(period={cur['period']}, k={cur['k']}) 本身分布 ──")
    wv = w_cur.dropna()
    print(f"  width: mean={wv.mean():.4f} median={wv.median():.4f} "
          f"p10={wv.quantile(.1):.4f} p90={wv.quantile(.9):.4f} max={wv.max():.4f}")

    # ── 5. 参数敏感性 ──
    print(f"\n── 参数敏感性对比（corr=与当前乘数序列的相关性, mu60=近60日均值, sat%=饱和占比）──")
    print(f"{'period':>7s} {'k':>5s} {'lb':>5s} | {'corr':>6s} {'mu60':>6s} {'sat%':>6s}")
    rows = []
    for period in sorted(set(ALT_PERIODS) | {cur["period"]}):
        for k in sorted(set(ALT_KS) | {cur["k"]}):
            for lb in sorted(set(ALT_LOOKBACKS) | {cur["lookback"]}):
                w = _bb_width(close, period, k)
                p = _width_pct(w, lb)
                m = _mult(p, cur["wmin"], cur["wmax"])
                both = pd.concat([m_cur, m], axis=1).dropna()
                corr = both.corr().iloc[0, 1] if len(both) > 2 else np.nan
                mu60 = m.tail(60).mean()
                sat = ((p <= 0.01) | (p >= 0.99)).sum() / max(p.count(), 1)
                tag = "  ← 当前" if (period, k, lb) == (cur["period"], cur["k"], cur["lookback"]) else ""
                rows.append((corr, period, k, lb, mu60, sat))
                print(f"{period:>7d} {k:>5.1f} {lb:>5d} | "
                      f"{corr:>6.3f} {mu60:>6.3f} {100*sat:>5.1f}%{tag}")

    rows_sorted = sorted(rows, key=lambda x: (x[0] is np.nan, x[0] if x[0] == x[0] else 0))
    print("  与当前序列相关性最低的组合（参数选择影响大的方向）:")
    for corr, period, k, lb, mu60, sat in rows_sorted[:5]:
        print(f"    period={period} k={k} lb={lb}: corr={corr:.3f}")

    # ── 6. lookback 影响 ──
    print(f"\n── lookback 影响（period={cur['period']}, k={cur['k']} 固定）──")
    for lb in ALT_LOOKBACKS:
        p = _width_pct(w_cur, lb)
        m = _mult(p, cur["wmin"], cur["wmax"])
        print(f"  lb={lb:>3d}: 近60日 mult 均值={m.tail(60).mean():.3f}, "
              f"最近5日={m.tail(5).round(3).tolist()}")

    # ── 7. 分年波动率结构 ──
    ret = close.pct_change()
    print(f"\n── 各年波动率（日收益率 std 年化）──")
    yr = ret.groupby(df["Date"].dt.year).std() * np.sqrt(252)
    for y, v in yr.items():
        print(f"  {y}: {v:.1%}")

    # ── 结论提示 ──
    print(f"\n── 诊断提示 ──")
    flags = []
    if sat_ratio > 0.2:
        flags.append(f"归一化饱和占比 {100*sat_ratio:.0f}% 偏高 → "
                     f"[min,max]=[{cur['wmin']},{cur['wmax']}] 映射区间或 lookback 可能不合适")
    if dmult.quantile(.95) > 0.15:
        flags.append(f"乘数日间跳动 p95={dmult.quantile(.95):.2f} 偏大 → "
                     f"网格频繁大幅缩放，可考虑加大 period 或 lookback")
    if not flags:
        flags.append("未发现明显异常，当前参数结构上合理；如需收益最优可跑 tune_bb_params.py")
    for f in flags:
        print(f"  • {f}")
    print("  注: 本脚本只做数据诊断，不跑回测、不评收益。")

    return m_cur


def main():
    parser = argparse.ArgumentParser(description="pm_bb_* 参数诊断（基于缓存日线）")
    parser.add_argument("--ts-code", type=str, required=True,
                        help="股票代码，如 03690.HK / 06181.HK")
    args = parser.parse_args()

    ts_code = args.ts_code
    common_config.init_config(ts_code)
    base_kwargs = dict(common_config.BASE_KWARGS)
    optimal_params = dict(common_config.OPTIMAL_PARAMS_ACTIVE)
    stock_name = base_kwargs.get("stock_name", ts_code)
    market = base_kwargs.get("market", "HK")

    # 用 BacktestConfig 合并基础参数+最优参数，解析出 pm_bb_* 默认值
    cfg = BacktestConfig(**{**base_kwargs, **optimal_params})

    print(f"{'='*60}")
    print(f"pm_bb_* 参数诊断 — {stock_name} {ts_code}")
    print(f"{'='*60}")

    df = cache_load(ts_code, market)
    if df is None or df.empty:
        print(f"  ❌ data_cache 中无 {ts_code} 缓存，请先运行 data_cache/update_cache.py")
        return
    df = df.sort_values("Date").reset_index(drop=True)
    print(f"缓存数据: {len(df)} 行, {df['Date'].min().date()} ~ {df['Date'].max().date()}")

    if not cfg.pm_bb_enabled:
        print("  ⚠️ 该标的 pm_bb_enabled=False，以下诊断仅供参考")

    _diagnose(cfg, df)


if __name__ == "__main__":
    main()
