#!/usr/bin/env python3
"""布林线通道参数优化 —— 基于流动性 TOP10 加仓做多回测

功能：
  1. 筛选近五年日均成交额最高的 10 只股票 (港股/A股)
  2. 对每只股票扫描布林线 (bb_period, bb_k) 参数组合
  3. 按四项核心指标综合评分排名：
     - 期望收益率 (20%)
     - 回归年度回报率 (25%)
     - R立方 (30%)
     - 稳健夏普比率 (25%)
  4. 输出最优参数建议

用法：
  python optimize_bollinger.py          # 默认港股
  python optimize_bollinger.py A        # A股

依赖：
  tushare, akshare, pandas, numpy
  环境变量 TUSHARE_TOKEN 需要已配置
"""

import os
import sys
import importlib
import warnings
from itertools import product
import pandas as pd
import numpy as np
import tushare as ts

warnings.filterwarnings("ignore")

from core.config import BacktestConfig
from core.data import fetch_data
from core.indicators import compute_indicators
from core.engine import run_backtest
from core.statistics import calc_stats

# ============================================================
# 配置
# ============================================================
MARKET = sys.argv[1].upper() if len(sys.argv) > 1 else "HK"
assert MARKET in ("HK", "A"), f"不支持的市场: {MARKET}"
os.environ["TURTLE_MARKET"] = MARKET

# 布林线参数扫描范围
BB_PERIODS = [10, 15, 20, 25, 30]
BB_K_VALUES = [1.0, 1.5, 2.0, 2.5, 3.0]

# 策略参数（固定）
MODE = "long_more"
MAX_PYRAMID = 5
ENTRY_PERIOD = 20
EXIT_PERIOD = 10
ATR_PERIOD = 20
N_VALUE = 2.0
RISK_PERCENT = 1.0
INITIAL_CAPITAL = 200000.0
YEARS = 8

_suffix = "a" if MARKET == "A" else "hk"
OUTPUT_CSV = os.path.join(os.path.dirname(__file__),
                          f"bollinger_param_analysis_{_suffix}.csv")


# ============================================================
# 工具函数
# ============================================================
def _load_ode_module():
    """加载 optimize_donchian_ema 模块"""
    mod_name = "optimize_donchian_ema"
    if mod_name in sys.modules:
        importlib.reload(sys.modules[mod_name])
    import optimize_donchian_ema as ode
    return ode


def _get_name_map(pro) -> dict[str, str]:
    """ts_code → 股票名称"""
    if MARKET == "A":
        df = pro.stock_basic(list_status="L", fields="ts_code,name")
        return dict(zip(df["ts_code"], df["name"]))
    else:
        df = pro.hk_basic()
        name_map = {}
        for _, r in df.iterrows():
            name_map[r["ts_code"]] = r["name"]
        return name_map


def _norm(series: pd.Series) -> pd.Series:
    """min-max 归一化"""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(0.5, index=series.index)
    return (series - mn) / (mx - mn)


def compute_composite_score(df: pd.DataFrame) -> pd.Series:
    """基于四项核心指标计算综合评分"""
    s_er = _norm(df["期望收益率"])
    s_rar = _norm(df["回归年度回报率 (%)"])
    s_r3 = _norm(df["R立方"])
    s_rs = _norm(df["稳健夏普比率"])
    return s_er * 0.20 + s_rar * 0.25 + s_r3 * 0.30 + s_rs * 0.25


# ============================================================
# 参数扫描
# ============================================================
def run_bb_parameter_scan(pro, top10_raw: list) -> pd.DataFrame:
    """对 TOP10 股票扫描布林线参数组合，返回综合评分 DataFrame"""
    # name_map = _get_name_map(pro)

    # 各参数组合的跨股票汇总
    combo_scores: dict[tuple, dict] = {}
    for period, k in product(BB_PERIODS, BB_K_VALUES):
        combo_scores[(period, k)] = {
            "期望收益率": [], "回归年度回报率 (%)": [],
            "R立方": [], "稳健夏普比率": [],
            "总收益率 (%)": [], "胜率 (%)": [], "最大回撤 (%)": [],
        }

    print(f"\n📊 扫描布林线参数: period × k = {len(BB_PERIODS)}×{len(BB_K_VALUES)}"
          f" = {len(BB_PERIODS) * len(BB_K_VALUES)} 组合 × {len(top10_raw)} 只股票")

    for rank, (ts_code, avg_amt, df_raw) in enumerate(top10_raw, 1):
        # name = name_map.get(ts_code, "")
        # print(f"\n[{rank}/{len(top10_raw)}] {ts_code} {name} ...")

        best_score = -999
        best_combo = None

        for period, k in product(BB_PERIODS, BB_K_VALUES):
            try:
                df = compute_indicators(
                    df_raw,
                    entry_period=ENTRY_PERIOD, exit_period=EXIT_PERIOD,
                    atr_period=ATR_PERIOD, channel_type="bollinger",
                    bb_period=period, bb_k=k,
                )
                config = BacktestConfig(
                    ts_code=ts_code, market=MARKET,
                    years=YEARS, mode=MODE, max_pyramid=MAX_PYRAMID,
                    entry_period=ENTRY_PERIOD, exit_period=EXIT_PERIOD,
                    atr_period=ATR_PERIOD, n_value=N_VALUE,
                    risk_percent=RISK_PERCENT, initial_capital=INITIAL_CAPITAL,
                    channel_type="bollinger", bb_period=period, bb_k=k,
                )
                trades, equity = run_backtest(df, config)
                stats = calc_stats(trades, config.initial_capital, equity)

                key = (period, k)
                combo_scores[key]["期望收益率"].append(stats["期望收益率"])
                combo_scores[key]["回归年度回报率 (%)"].append(stats["回归年度回报率 (%)"])
                combo_scores[key]["R立方"].append(stats["R立方"])
                combo_scores[key]["稳健夏普比率"].append(stats["稳健夏普比率"])
                combo_scores[key]["总收益率 (%)"].append(stats["总收益率 (%)"])
                combo_scores[key]["胜率 (%)"].append(stats["胜率 (%)"])
                combo_scores[key]["最大回撤 (%)"].append(stats["最大回撤 (%)"])

                # 计算该组合对这只股票的评分
                single_df = pd.DataFrame([stats])
                score = compute_composite_score(single_df).iloc[0]
                if score > best_score:
                    best_score = score
                    best_combo = (period, k)

            except Exception as e:
                pass  # 单个参数组合失败不影响其他

        if best_combo:
            print(f"  ✅ 最佳: bb_period={best_combo[0]}, bb_k={best_combo[1]}  "
                  f"(评分: {best_score:.4f})")

    # ---- 跨股票汇总 ----
    summary = []
    for (period, k), scores in combo_scores.items():
        avg = {key: np.mean(vals) if vals else 0
               for key, vals in scores.items()}
        summary.append({
            "bb_period": period,
            "bb_k": k,
            "avg_期望收益率": round(avg["期望收益率"], 4),
            "avg_回归年度回报率": round(avg["回归年度回报率 (%)"], 2),
            "avg_R立方": round(avg["R立方"], 2),
            "avg_稳健夏普比率": round(avg["稳健夏普比率"], 2),
            "avg_总收益率": round(avg["总收益率 (%)"], 2),
            "avg_胜率": round(avg["胜率 (%)"], 1),
            "avg_最大回撤": round(avg["最大回撤 (%)"], 2),
            "样本数": len(scores["期望收益率"]),
        })

    summary_df = pd.DataFrame(summary)

    # ---- 综合评分 ----
    score_df = pd.DataFrame({
        "期望收益率": summary_df["avg_期望收益率"],
        "回归年度回报率 (%)": summary_df["avg_回归年度回报率"],
        "R立方": summary_df["avg_R立方"],
        "稳健夏普比率": summary_df["avg_稳健夏普比率"],
    })
    summary_df["综合得分"] = compute_composite_score(score_df)

    summary_df = summary_df.sort_values("综合得分", ascending=False).reset_index(drop=True)
    summary_df["排名"] = range(1, len(summary_df) + 1)
    return summary_df


# ============================================================
# 主流程
# ============================================================
def main():
    token = os.environ.get("TUSHARE_TOKEN")
    if not token:
        print("❌ 请先设置环境变量 TUSHARE_TOKEN")
        return
    pro = ts.pro_api(token)

    label = "A股" if MARKET == "A" else "港股"
    currency = "CNY" if MARKET == "A" else "HKD"

    # ---- 1. 获取日均成交额>1亿 股票 ----
    ode = _load_ode_module()
    print("=" * 80)
    print(f"🏆 近一年{label}日均成交额 >1亿")
    print("=" * 80)
    top10_raw = ode.fetch_liquidity(pro, ode.START_DATE, ode.END_DATE)

    name_map = _get_name_map(pro)
    for i, (code, amt, _) in enumerate(top10_raw, 1):
        name = name_map.get(code, "")
        print(f"  #{i} {code} {name}  日均成交额: {amt:>15,.0f} {currency}")

    # ---- 2. 参数扫描 ----
    print(f"\n{'=' * 80}")
    print(f"📊 布林线参数扫描 ({label} 加仓做多)")
    print(f"   bb_period: {BB_PERIODS}")
    print(f"   bb_k: {BB_K_VALUES}")
    print(f"   策略: mode={MODE}, max_pyramid={MAX_PYRAMID}, years={YEARS}")
    print(f"{'=' * 80}")

    summary = run_bb_parameter_scan(pro, top10_raw)

    # ---- 3. 输出排名 ----
    print(f"\n{'=' * 100}")
    print("🏆 布林线参数综合排名")
    print(f"   评分权重: 期望收益率20% + 回归年度回报率25% + R立方30% + 稳健夏普比率25%")
    print(f"{'=' * 100}")
    cols = ["排名", "bb_period", "bb_k", "综合得分",
            "avg_期望收益率", "avg_回归年度回报率", "avg_R立方",
            "avg_稳健夏普比率", "avg_总收益率", "avg_胜率",
            "avg_最大回撤", "样本数"]
    print(summary[cols].to_string(index=False))

    # ---- 4. 最优建议 ----
    best = summary.iloc[0]
    print(f"\n{'=' * 80}")
    print("🎯 布林线最优参数建议")
    print(f"{'=' * 80}")
    print(f"  推荐 bb_period = {best['bb_period']}, bb_k = {best['bb_k']}")
    print(f"  综合得分: {best['综合得分']:.4f}")
    print(f"  平均期望收益率: {best['avg_期望收益率']:.4f}")
    print(f"  平均回归年度回报率: {best['avg_回归年度回报率']:.2f}%")
    print(f"  平均R立方: {best['avg_R立方']:.2f}")
    print(f"  平均稳健夏普比率: {best['avg_稳健夏普比率']:.2f}")
    print(f"  平均总收益率: {best['avg_总收益率']:.2f}%")
    print(f"  平均胜率: {best['avg_胜率']:.1f}%")
    print(f"  平均最大回撤: {best['avg_最大回撤']:.2f}%")

    # ---- 5. 分组统计 ----
    print(f"\n{'=' * 80}")
    print("📊 按 bb_period 分组 (平均综合得分)")
    print(f"{'=' * 80}")
    for period in BB_PERIODS:
        sub = summary[summary["bb_period"] == period]
        if sub.empty:
            continue
        top3_k = sub.head(3)["bb_k"].tolist()
        print(f"  period={period:<4} 组合:{len(sub):>2}  "
              f"均分:{sub['综合得分'].mean():.4f}  "
              f"Top k值:{top3_k}")

    print(f"\n{'=' * 80}")
    print("📊 按 bb_k 分组 (平均综合得分)")
    print(f"{'=' * 80}")
    for k in BB_K_VALUES:
        sub = summary[summary["bb_k"] == k]
        if sub.empty:
            continue
        top3_p = sub.head(3)["bb_period"].tolist()
        print(f"  k={k:<5} 组合:{len(sub):>2}  "
              f"均分:{sub['综合得分'].mean():.4f}  "
              f"Top period:{top3_p}")

    # ---- 6. 保存 ----
    summary.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n✅ 结果已保存至 {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
