#!/usr/bin/env python3
"""海龟交易法则 第十一章 最优化实验

重现 Curtis Faith《The Way of the Turtle》第十一章的实验逻辑：
  Part 1: 全量搜索 —— 全局最优 → 更新 DEFAULT_PARAMS
  Part 2: 单参数扫描 —— 以全局最优为起点，验证山峰形态 & 峭壁检测
  Part 3: 滚动窗口 + 模拟数据 —— 跨时段稳定性 + 未来情景压力测试

用法：
  python turtle_optimize.py
"""

import itertools
import sys
import os
import time
import copy
from datetime import timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# 中文字体配置 (macOS)
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti SC', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.config import BacktestConfig
from core.data import fetch_data
from core.indicators import compute_indicators
from core.engine import run_backtest
from core.statistics import calc_stats


# ═══════════════════════════════════════════════════════════════
# 基础配置（与 grid_search_pm.py 一致）
# ═══════════════════════════════════════════════════════════════

BASE_KWARGS = dict(
    ts_code="09992.HK",
    stock_name="泡泡玛特",
    market="HK",
    mode="position_management_1",
    per_shares=200,
    start_date="20201211",
    pm_start_date="20210112", #全量数据所有指标均不为空的那一天开始
    user_dmi=True,
    pm_adaptive=True,
    pm_adaptive_grid=False,
    pm_dynamic_position=False,
    ema_fast=0,
    ema_slow=0,
    pm_initial_shares=2,
    initial_capital=40000,
    data_source="tushare",
)
# BASE_KWARGS = dict(
#     ts_code="02015.HK",
#     stock_name="理想",
#     market="HK",
#     mode="position_management_1",
#     per_shares=100,
#     pm_start_date="20220913",
#     start_date="20220813",
#     user_dmi=True,
#     pm_adaptive=True,
#     pm_adaptive_grid=False,
#     pm_dynamic_position=False,
#     ema_fast=0,
#     ema_slow=0,
#     pm_initial_shares=25,
#     initial_capital=0,
#     data_source="akshare",
# )
# BASE_KWARGS = dict(
#         ts_code="03690.HK",
#         stock_name="美团",
#         market="HK",
#         mode="position_management_1",
#         per_shares=100,
#         pm_start_date="20181023",
#         start_date="20180920",
#         user_dmi=True,
#         pm_adaptive=True,
#         pm_adaptive_grid=False,
#         pm_dynamic_position=False,
#         ema_fast=0,
#         ema_slow=0,
#         pm_initial_shares=2,
#         initial_capital=0,
#         data_source="akshare",
#     )
# BASE_KWARGS = dict(
#         ts_code="06181.HK",
#         stock_name="老铺",
#         market="HK",
#         mode="position_management_1",
#         per_shares=100,
#         pm_start_date="20240729",
#         start_date="20240628",
#         user_dmi=True,
#         pm_adaptive=True,
#         pm_adaptive_grid=False,
#         pm_dynamic_position=False,
#         ema_fast=0,
#         ema_slow=0,
#         pm_initial_shares=3,
#         initial_capital=40000,
#         data_source="akshare",
#     )
# 与 grid_search_pm.py 完全一致的搜索空间
SWEEP_RANGES = {
     "pm_agg_k": [1.0], #0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
    "pm_add_pct": [0.25, 0.30, 0.35, 0.40, 0.50],
    "pm_reduce_pct": [0.25, 0.35, 0.60, 0.75],
    "pm_grid_up": [0.03, 0.04, 0.05, 0.07, 0.10],
    "pm_grid_down": [0.03, 0.05, 0.07, 0.10, 0.12],
    "pm_adx_threshold": [25,30,35,40],
}

PARAM_LABELS = {
    "pm_grid_down":       "网格向下间距 (gd)",
    "pm_grid_up":         "网格向上间距 (gu)",
    "pm_reduce_pct":      "减仓比例 (rp)",
    "pm_add_pct":         "加仓比例 (ap)",
    "pm_agg_k":           "激进倍数 (k)",
    "pm_adx_threshold":    "强势值(adx)"
}

# 单参数扫描步长：adx 整数步进，其余 0.01
SWEEP_STEP = {
    "pm_adx_threshold": 1,
}

# 默认参数（需满足 gd>gu, rp>ap 约束）
DEFAULT_PARAMS = dict(
    pm_grid_down=0.12,
    pm_grid_up=0.1,
    pm_reduce_pct=0.75,
    pm_add_pct=0.5,
    pm_adx_threshold=25,
    pm_agg_k=0.8,
)

# 参数边界（防止生成不合法的值）
_PARAM_BOUNDS = {
    "pm_grid_down": (0.05, 0.35), "pm_grid_up": (0.05, 0.35),
    "pm_reduce_pct": (0.60, 0.90), "pm_add_pct": (0.20, 0.50),
    "pm_agg_k": (0.4, 1.0),"pm_adx_threshold":(25,55),
}
# ═══════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════

def _constraint_ok(params: dict) -> bool:
    gd = params.get("pm_grid_down", 0)
    gu = params.get("pm_grid_up", 0)
    rp = params.get("pm_reduce_pct", 0)
    ap = params.get("pm_add_pct", 0)
    return gd > gu and rp > ap


def _minmax_norm(raw: list[float]) -> list[float]:
    """Min-max 归一化到 0-100，所有值相等时返回 50"""
    lo, hi = min(raw), max(raw)
    if hi - lo < 1e-12:
        return [50.0] * len(raw)
    return [(v - lo) / (hi - lo) * 100 for v in raw]


def _plot_sweep_curves(sweeps: dict, out_dir: str, filename: str = "turtle_sweeps.png",
                        title: str = "单参数扫描 — Score & 分项指标曲线",
                        base_params: dict | None = None):
    """绘制每个参数的值-vs-指标曲线（score + R³ + 稳健夏普 + 超额收益，共用归一化 y 轴）"""
    param_names = [k for k in sweeps.keys() if not k.startswith("_")]
    n = len(param_names)
    if n == 0:
        return

    lines_cfg = [
        ("score",         "Score (综合)",       "steelblue",  "-",  "o"),
        ("r_cubed",       "R³",                 "darkorange", "--", "s"),
        ("robust_sharpe", "稳健夏普",            "seagreen",   "-.", "D"),
        ("excess",        "超额收益",            "firebrick",  ":",  "^"),
    ]

    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, pname in enumerate(param_names):
        ax = axes[i]
        results = sweeps[pname]
        label = PARAM_LABELS.get(pname, pname)

        # 按 value 排序
        results_sorted = sorted(results, key=lambda r: r["value"])
        x = [r["value"] for r in results_sorted]

        for key, disp_name, color, ls, marker in lines_cfg:
            raw = [r[key] for r in results_sorted]
            y = _minmax_norm(raw)
            ax.plot(x, y, color=color, linestyle=ls, marker=marker,
                    linewidth=1.5, markersize=5, label=disp_name)

        # 标注 score 最高点（score 已 min-max 归一化到 0-100）
        best = max(results_sorted, key=lambda r: r["score"])
        scores_norm = _minmax_norm([r["score"] for r in results_sorted])
        best_idx = next(j for j, r in enumerate(results_sorted) if r is best)
        ax.annotate(f"{best['value']:.3f}",
                    xy=(best["value"], scores_norm[best_idx]),
                    xytext=(0, -12), textcoords="offset points",
                    fontsize=8, color="darkred", ha="center",
                    arrowprops=dict(arrowstyle="->", color="darkred", lw=0.8))

        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel("归一化得分 (0–100)", fontsize=11)
        ax.set_title(f"{label}", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")

        # 标注固定参数（除当前扫描参数外）
        if base_params:
            param_lines = []
            for pn in ["pm_grid_down", "pm_grid_up", "pm_reduce_pct",
                        "pm_add_pct", "pm_agg_k", "pm_adx_threshold"]:
                plabel = PARAM_LABELS.get(pn, pn)
                # 缩短短标签用于图内标注
                short = plabel.split("(")[-1].rstrip(")") if "(" in plabel else plabel[-4:]
                if pn == pname:
                    param_lines.append(f"{short}={base_params[pn]:.3f} ←")
                else:
                    param_lines.append(f"{short}={base_params[pn]:.3f}")
            # 两列排版
            left_col = "  ".join(param_lines[:3])
            right_col = "  ".join(param_lines[3:])
            anno_text = f"{left_col}\n{right_col}"
            ax.text(0.02, 0.97, anno_text, transform=ax.transAxes,
                    fontsize=6, fontfamily="monospace", color="gray",
                    va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor="lightgray", alpha=0.85))

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()

    png_path = os.path.join(out_dir, filename)
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"📈 参数扫描曲线已保存: {png_path}")


def _single_round_sweep(df_base: pd.DataFrame, base_params: dict,
                         benchmark_return: float) -> dict:
    """以 base_params 为固定基准，对每个参数做一轮独立单参数扫描（不更新参数）。
    返回与 part2_param_sweep 中 all_sweeps 相同结构的 dict。"""
    sweeps = {}
    for pname in SWEEP_RANGES:
        label = PARAM_LABELS[pname]
        current_val = base_params[pname]
        lo, hi = _PARAM_BOUNDS[pname]
        low = lo
        high = hi
        step = SWEEP_STEP.get(pname, 0.01)
        sweep_vals = np.round(np.arange(low, high + step * 0.01, step), 3).tolist()
        if current_val not in sweep_vals:
            sweep_vals = sorted(set(sweep_vals + [current_val]))
        batch = []
        skipped_constraint = []
        skipped_zero_trades = []
        for val in sweep_vals:
            params = copy.deepcopy(base_params)
            params[pname] = val
            if not _constraint_ok(params):
                skipped_constraint.append(val)
                continue
            stats = run_one_backtest(params, BASE_KWARGS, df_base,
                                     benchmark_return=benchmark_return)
            if stats:
                stats["_sweep_val"] = val
                batch.append(stats)
            else:
                skipped_zero_trades.append(val)

        if skipped_constraint:
            print(f"     ⛔ 约束跳过 ({label}): {skipped_constraint}")
        if skipped_zero_trades:
            print(f"     ⚠️ 0交易跳过 ({label}): {skipped_zero_trades}")

        if len(batch) < 2:
            continue

        compute_percentile_scores(batch)

        results = []
        for s in batch:
            results.append({
                "value": s["_sweep_val"],
                "score": s["score"],
                "CAGR": round(s["CAGR (%)"], 2),
                "MaxDD": round(s["最大回撤 (%)"], 2),
                "excess": round(s["超额收益 (%)"], 2),
                "robust_sharpe": round(s["稳健夏普比率"], 2),
                "r_cubed": round(s["R立方"], 2),
                "trades": s["总交易"],
            })

        sweeps[pname] = results
        best = max(results, key=lambda x: x["score"])
        print(f"     🏔️ {label} 峰值: {best['value']:.3f}  (得分={best['score']:.2f})")

    sweeps["_base_params"] = base_params
    return sweeps


def run_one_backtest(params: dict, base_kwargs: dict, df: pd.DataFrame,
                     benchmark_return: float = 0.0) -> dict | None:
    if not _constraint_ok(params):
        return None
    try:
        config = BacktestConfig(**{**base_kwargs, **params})
        trades, equity, eq_dates = run_backtest(df, config)
        stats = calc_stats(trades, config.initial_capital, equity, eq_dates,
                           benchmark_return=benchmark_return)
        if stats["总交易"] == 0:
            return None
        return _map_stats_keys(stats)
    except Exception:
        return None


def composite_score(stats: dict) -> float:
    """多指标综合评分（与 grid_search_pm.py 一致）。

    超额收益40% + 稳健夏普40% + R³10%。
    """
    r3 = min(stats.get("R立方", 0.0) or 0.0, 10.0)
    rs = stats.get("稳健夏普比率", 0.0) or 0.0
    excess = stats.get("超额收益 (%)", 0.0) or 0.0
    return max(0.0, min(100.0,
               (excess + 20) / 60 * 53.9 + (rs + 2) / 8 * 22.5 + r3 / 10 * 23.6))


def compute_percentile_scores(results: list[dict]) -> list[dict]:
    """批量百分位归一化评分（与 grid_search_pm.py 完全一致）。

    四项指标百分位加权，超额收益为主，完全免疫极端值。
    每个结果会原地增加 'score' 字段。
    """
    if len(results) < 3:
        for r in results:
            r["score"] = 50.0
        return results

    df = pd.DataFrame(results)
    weights = {"excess": 0.50, "robust_sharpe": 0.40, "r_cubed": 0.10}

    for metric, w in weights.items():
        pcts = df[metric].rank(pct=True) * 100
        for i, r in enumerate(results):
            r[f"pct_{metric}"] = round(pcts.iloc[i], 1)

    for r in results:
        score = sum(r.get(f"pct_{m}", 50) * w for m, w in weights.items())
        r["score"] = round(score, 2)

    return results


def _map_stats_keys(stats: dict) -> dict:
    """将 calc_stats 返回的中文 key 映射为 compute_percentile_scores 需要的英文 key。"""
    stats.setdefault("excess", stats.get("超额收益 (%)", 0.0) or 0.0)
    stats.setdefault("robust_sharpe", stats.get("稳健夏普比率", 0.0) or 0.0)
    stats.setdefault("r_cubed", stats.get("R立方", 0.0) or 0.0)
    return stats


def _stats_from_all_stats_row(row: dict) -> dict:
    """从 Part 1 all_stats 行中提取 compute_percentile_scores 需要的原始指标。"""
    return {
        "excess": row.get("excess", 0.0) or 0.0,
        "robust_sharpe": row.get("robust_sharpe", 0.0) or 0.0,
        "r_cubed": row.get("r_cubed", 0.0) or 0.0,
    }


def compute_buy_hold(df: pd.DataFrame, start_date: str) -> float:
    mask = df["Date"] >= start_date
    close = df.loc[mask, "Close"]
    if len(close) < 2:
        return 0.0
    return (close.iloc[-1] / close.iloc[0] - 1) * 100


# ═══════════════════════════════════════════════════════════════
# Part 2: 单参数扫描
# ═══════════════════════════════════════════════════════════════

def part2_param_sweep(df_base: pd.DataFrame):
    """Part 2: 单参数扫描 —— 以全局最优为起点，验证山峰形态 & 峭壁检测。

    串行贪心：扫完一个参数 → 取最优值更新 → 扫下一个。
    """
    print("=" * 70)
    print("🐢 Part 2: 单参数扫描 —— 山峰曲线 & 峭壁检测")
    print("=" * 70)
    print("  方法：串行贪心 — 每扫完一个参数，取最优值更新基值再扫下一个\n")

    bm = compute_buy_hold(df_base, BASE_KWARGS["pm_start_date"])
    all_sweeps = {}

    # 使用可变副本，扫完一个参数就更新
    current_params = copy.deepcopy(DEFAULT_PARAMS)

    while True:
        last_param = copy.deepcopy(current_params)
        for pname in SWEEP_RANGES:
            label = PARAM_LABELS[pname]
            current_val = current_params[pname]
            lo, hi = _PARAM_BOUNDS[pname]
            # 从当前值向两端扩展 ±50%，生成30个候选值
            low = max(lo, current_val * 0.5)
            high = min(hi, current_val * 1.5)
            print(f"{pname}:{low},{high}")
            sweep_vals = np.round(np.linspace(low, high, 30), 3).tolist()
            # 确保当前值在列表中
            if current_val not in sweep_vals:
                sweep_vals = sorted(set(sweep_vals + [current_val]))

            pvalues = sweep_vals
            print(f"  📐 扫描 {label}: {low:.3f}→{high:.3f} ({len(pvalues)}个值)")
            print(f"     当前基值: " + ", ".join(f"{PARAM_LABELS.get(k, k)}={v}"
                                                 for k, v in current_params.items() if k in SWEEP_RANGES))

            # 收集该参数所有值的回测统计
            batch = []
            for val in pvalues:
                params = copy.deepcopy(current_params)
                params[pname] = val
                stats = run_one_backtest(params, BASE_KWARGS, df_base, benchmark_return=bm)
                if stats:
                    stats["_sweep_val"] = val
                    batch.append(stats)

            if len(batch) < 2:
                print(f"     ⚠️ 有效结果不足\n")
                continue

            # 批量百分位归一化
            compute_percentile_scores(batch)

            results = []
            for s in batch:
                results.append({
                    "value": s["_sweep_val"],
                    "score": s["score"],
                    "CAGR": round(s["CAGR (%)"], 2),
                    "MaxDD": round(s["最大回撤 (%)"], 2),
                    "excess": round(s["超额收益 (%)"], 2),
                    "robust_sharpe": round(s["稳健夏普比率"], 2),
                    "r_cubed": round(s["R立方"], 2),
                    "trades": s["总交易"],
                })

            all_sweeps[pname] = results

            # 山峰
            best = max(results, key=lambda x: x["score"])
            worst = min(results, key=lambda x: x["score"])
            print(f"     百分位得分范围: {worst['score']:.2f} ~ {best['score']:.2f}")
            print(f"     🏔️ 峰值: {label} = {best['value']:.3f}  "
                  f"(得分={best['score']:.2f}, CAGR={best['CAGR']:.1f}% / DD={best['MaxDD']:.1f}%)")

            # 更新基值
            old_val = current_params[pname]
            if old_val != best["value"]:
                current_params[pname] = best["value"]
                print(f"     📌 更新: {label} {old_val:.3f} → {best['value']:.3f}")

            # 峭壁检测：对每个原始指标单独检验峰值附近的波动
            peak_val = best["value"]
            near_peak = [s for s in batch
                         if abs(s["_sweep_val"] - peak_val) / max(peak_val, 0.001) <= 0.25]
            if len(near_peak) >= 2:
                metrics = {
                    "超额收益": [s.get("excess", 0) for s in near_peak],
                    "稳健夏普": [s.get("robust_sharpe", 0) for s in near_peak],
                    "R³": [min(s.get("r_cubed", 0), 100) for s in near_peak],
                }
                cliff_tags = []
                for mname, mvals in metrics.items():
                    mrange = max(mvals) - min(mvals)
                    mavg = np.mean(mvals)
                    cr = abs(mrange / max(abs(mavg), 0.01)) if abs(mavg) > 0.01 else 0
                    if cr > 0.5:
                        cliff_tags.append(f"{mname} ⚠️{cr:.0%}")
                    elif cr > 0.25:
                        cliff_tags.append(f"{mname} ⚡{cr:.0%}")
                    else:
                        cliff_tags.append(f"{mname} ✓{cr:.0%}")
                print(f"     峭壁检测: " + "  ".join(cliff_tags))
            print()
        if current_params == last_param:
            break

    # 最终串行贪心结果
    print("  ── 串行贪心最终参数 ──")
    second_best_candidates = []  # 每个参数峰值两侧的第二优参数
    for pname in SWEEP_RANGES:
        label = PARAM_LABELS.get(pname, pname)
        orig = DEFAULT_PARAMS[pname]
        curr = current_params[pname]
        arrow = "→" if orig != curr else " (未变)"
        print(f"  {label:<18} {orig:.3f} {arrow} {curr:.3f}")

        # 收集该参数扫描中峰值两侧的第二优值
        sweep_results = all_sweeps.get(pname, [])
        if len(sweep_results) >= 3:
            sorted_r = sorted(sweep_results, key=lambda x: x["score"], reverse=True)
            peak_val = sorted_r[0]["value"]
            left_vals = [r for r in sorted_r if r["value"] < peak_val]
            right_vals = [r for r in sorted_r if r["value"] > peak_val]
            for side_vals, side_tag in [(left_vals, "左"), (right_vals, "右")]:
                if side_vals:
                    second_val = side_vals[0]["value"]  # 该侧得分最高的
                    variant = copy.deepcopy(current_params)
                    variant[pname] = second_val
                    if _constraint_ok(variant):
                        second_best_candidates.append(
                            (f"{label}@{second_val:.3f}", variant))
                        print(f"         {side_tag}侧第二优: {second_val:.3f} (得分={side_vals[0]['score']:.1f})")
    print()

    return all_sweeps, current_params, second_best_candidates


# ═══════════════════════════════════════════════════════════════
# Part 3: 滚动窗口检验 + 模拟数据样本外
# ═══════════════════════════════════════════════════════════════

def part3_rolling_sim(df_base: pd.DataFrame,
                      greedy_params: dict | None = None,
                      second_best_candidates: list | None = None,
                      part1_all_stats: list | None = None):
    """Part 3: 滚动窗口检验 + 模拟数据样本外。

    样本内：9个月窗口 × 20天步长，评估参数跨时段稳定性。
    样本外：8种模拟行情压力测试。
    候选参数 = 贪心最优 + Part 2 峰值两侧第二优参数。

    评分：滚动窗口和模拟数据统一做跨组百分位归一化。
    综合得分 = Part1全量得分 × 1/3 + 滚动均分 × 1/3 + 模拟加权 × 1/3。
    P10 仅用于稳定性参考，不参与综合得分。
    """
    print("\n" + "=" * 70)
    print("🐢 Part 3: 滚动窗口检验 + 模拟数据样本外")
    print("=" * 70)

    dates = pd.to_datetime(df_base["Date"])
    n = len(df_base)
    WINDOW_BARS = 189
    STEP_BARS = 20
    WARMUP = 60

    n_windows = max(0, (n - WARMUP - WINDOW_BARS) // STEP_BARS + 1)
    print(f"  数据: {n}条, 窗口: {WINDOW_BARS}天 × 步长{STEP_BARS}天 → ~{n_windows}个窗口")
    print(f"  样本外: 8种模拟行情（252天未来路径）")
    print(f"  评分: 跨候选统一百分位归一化，P10仅作稳定性参考\n")

    # ── 候选参数：贪心最优 + 第二优参数 ──
    candidates = []
    if greedy_params:
        candidates.append(("贪心最优", copy.deepcopy(greedy_params)))
    if second_best_candidates:
        for label, p in second_best_candidates:
            candidates.append((label, p))

    # 去重
    seen = set()
    unique = []
    for label, p in candidates:
        key = (p["pm_grid_down"], p["pm_grid_up"], p["pm_reduce_pct"],
               p["pm_add_pct"], p["pm_agg_k"],p["pm_adx_threshold"])
        if key not in seen:
            seen.add(key)
            unique.append((label, p))
    candidates = unique
    print(f"  候选参数: {len(candidates)} 组\n")

    if not candidates:
        print("  ⚠️ 无候选参数\n")
        return None

    # ── Part 1 全量得分：取原始指标，在候选之间重新跨组归一化 ──
    # Part 1 内部的百分位得分不能直接用（基准组不同），需要取出 R³/超额/夏普
    # 原始值，在当前候选之间重新跑 compute_percentile_scores。
    part1_scores = {}
    if part1_all_stats:
        # 构建精确匹配查找表：参数6元组 → 原始指标
        p1_lookup = {}
        for r in part1_all_stats:
            key = (r.get("pm_grid_down"), r.get("pm_grid_up"),
                   r.get("pm_reduce_pct"), r.get("pm_add_pct"),
                   r.get("pm_agg_k"), r.get("pm_adx_threshold", 25))
            p1_lookup[key] = _stats_from_all_stats_row(r)

        # 收集每个候选的 Part 1 原始指标
        cand_p1_stats = []
        cand_p1_labels = []
        for label, params in candidates:
            key = (params["pm_grid_down"], params["pm_grid_up"],
                   params["pm_reduce_pct"], params["pm_add_pct"],
                   params["pm_agg_k"], params.get("pm_adx_threshold", 25))
            if key in p1_lookup:
                row = dict(p1_lookup[key])
                row["_cand_label"] = label
                cand_p1_stats.append(row)
                cand_p1_labels.append(label)
            else:
                # 参数不在 Part 1 网格中（如 adx 被 Part 2 优化为其他值），
                # 单独跑一次全量回测取原始指标
                bm = compute_buy_hold(df_base, BASE_KWARGS["pm_start_date"])
                stats = run_one_backtest(params, BASE_KWARGS, df_base, benchmark_return=bm)
                if stats:
                    row = {
                        "excess": stats.get("excess", 0.0) or 0.0,
                        "robust_sharpe": stats.get("robust_sharpe", 0.0) or 0.0,
                        "r_cubed": stats.get("r_cubed", 0.0) or 0.0,
                        "_cand_label": label,
                    }
                    cand_p1_stats.append(row)
                    cand_p1_labels.append(label)

        # 在候选之间重新跨组百分位归一化
        if len(cand_p1_stats) >= 3:
            compute_percentile_scores(cand_p1_stats)
            for s in cand_p1_stats:
                part1_scores[s["_cand_label"]] = s["score"]
        elif len(cand_p1_stats) > 0:
            for s in cand_p1_stats:
                part1_scores[s["_cand_label"]] = 50.0

    # 没有 Part 1 数据的候选给中性分
    for label, _ in candidates:
        if label not in part1_scores:
            part1_scores[label] = 50.0

    # ── 样本内：滚动窗口评估 ──
    print("  ═══ 样本内：滚动窗口评估 ═══")
    t0 = time.time()

    # 收集所有候选×所有窗口的 stats，统一做跨组百分位
    all_window_stats = []  # [{stats, candidate_label, window_idx}]
    candidate_window_scores = {}  # {label: [pct_score_w1, ...]}

    for label, params in candidates:
        candidate_window_scores[label] = []
        win_count = 0
        for start_idx in range(WARMUP, n - WINDOW_BARS, STEP_BARS):
            end_idx = start_idx + WINDOW_BARS
            warmup_start = max(0, start_idx - WARMUP)
            df_win = df_base.iloc[warmup_start:end_idx].copy()
            ws = pd.Timestamp(dates[start_idx]).strftime("%Y%m%d")

            win_mask = df_win["Date"] >= ws
            win_close = df_win.loc[win_mask, "Close"]
            bm = ((win_close.iloc[-1] / win_close.iloc[0] - 1) * 100
                  if len(win_close) >= 2 else 0.0)

            win_kwargs = {**BASE_KWARGS, "pm_start_date": ws}
            stats = run_one_backtest(params, win_kwargs, df_win, benchmark_return=bm)
            if stats and stats.get("总交易", 0) > 0:
                stats["_cand_label"] = label
                all_window_stats.append(stats)
            win_count += 1

    # 统一百分位归一化（跨候选、跨窗口）
    if len(all_window_stats) >= 3:
        compute_percentile_scores(all_window_stats)
        for s in all_window_stats:
            candidate_window_scores[s["_cand_label"]].append(s["score"])
    else:
        print("  ⚠️ 有效窗口统计不足\n")
        return None

    # 稳定性阈值 = 所有候选 P10 的中位数
    all_p10 = []
    for label, _ in candidates:
        sc = candidate_window_scores.get(label, [])
        if len(sc) >= 10:
            all_p10.append(np.percentile(sc, 10))
    p10_threshold = np.median(all_p10) if all_p10 else 30

    print(f"  P10稳定性阈值(中位数): {p10_threshold:.1f}")
    print(f"  {'候选':<14} {'参数':<45} {'有效':>4} {'均分':>6} {'P10':>6} {'稳定性'}")
    print(f"  {'─' * 85}")
    for label, params in candidates:
        scores = candidate_window_scores.get(label, [])
        n_win = len(scores)
        if n_win >= 10:
            p10 = np.percentile(scores, 10)
            avg = np.mean(scores)
            p_str = f"gd={params['pm_grid_down']:.2f} gu={params['pm_grid_up']:.2f} " \
                    f"rp={params['pm_reduce_pct']:.2f} ap={params['pm_add_pct']:.2f} " \
                    f"k={params['pm_agg_k']:.2f} adx={params['pm_adx_threshold']:.2f}"
            stable = "✅ 稳定" if p10 > p10_threshold else "⚠️ 偏弱"
            print(f"  {label:<14} {p_str:<45} {n_win:>4} {avg:>6.1f} {p10:>6.1f} {stable}")
        else:
            print(f"  {label:<14} 有效窗口不足 ({n_win})")

    # 每个候选取所有窗口百分位得分的均值作为滚动窗口代表分
    rolling_scores = {}
    for label in [c[0] for c in candidates]:
        scores = candidate_window_scores.get(label, [])
        rolling_scores[label] = float(np.mean(scores)) if len(scores) >= 10 else 0.0

    # 构建每窗口明细（用于 Excel 导出）
    window_details = []
    for s in all_window_stats:
        window_details.append({
            "candidate": s["_cand_label"],
            "score": s["score"],
            "pct_excess": s.get("pct_excess", float("nan")),
            "pct_robust_sharpe": s.get("pct_robust_sharpe", float("nan")),
            "pct_r_cubed": s.get("pct_r_cubed", float("nan")),
            "CAGR": round(s["CAGR (%)"], 2),
            "MaxDD": round(s["最大回撤 (%)"], 2),
            "excess": round(s.get("excess", 0), 2),
            "robust_sharpe": round(s.get("robust_sharpe", 0), 2),
            "r_cubed": round(s.get("r_cubed", 0), 2),
            "trades": s["总交易"],
        })

    print(f"  样本内耗时: {(time.time()-t0)/60:.1f} 分钟\n")

    # ── 样本外：模拟数据 ──
    print("  ═══ 样本外：8种模拟行情 ═══")
    from simulate_market_scenarios import generate_ohlcv, get_last_price as sim_get_last_price
    from core.cache import load as cache_load

    try:
        last_price, last_date = sim_get_last_price()
    except Exception:
        cached = cache_load(BASE_KWARGS["ts_code"], BASE_KWARGS["market"])
        last_price = float(cached["Close"].iloc[-1])
        last_date = cached["Date"].iloc[-1]

    sim_start = (last_date + timedelta(days=1)).strftime("%Y%m%d")
    sim_days = 252
    print(f"  模拟起始: {sim_start}, {sim_days}个交易日\n")

    SCENARIO_LABEL_MAP = {
        "单边上涨": "up", "单边下跌": "down", "震荡向上": "volatile_up",
        "震荡向下": "volatile_down", "横盘震荡": "sideways",
        "V形反转": "v_reversal", "倒V反转": "inv_v_reversal", "缓慢阴跌": "slow_grind_down",
    }
    SCENARIO_PROBS = {
        "单边上涨": 0.02, "单边下跌": 0.05, "震荡向上": 0.10, "震荡向下": 0.20,
        "横盘震荡": 0.15, "V形反转": 0.10, "倒V反转": 0.03, "缓慢阴跌": 0.35,
    }

    # 生成模拟数据
    cached = cache_load(BASE_KWARGS["ts_code"], BASE_KWARGS["market"])
    sim_dfs = {}
    sim_benchmarks = {}
    for label, trend_type in SCENARIO_LABEL_MAP.items():
        df_sim = generate_ohlcv(sim_start, sim_days, last_price, trend_type,
                                ts_code=BASE_KWARGS["ts_code"], market=BASE_KWARGS["market"])
        s_start = df_sim["Date"].min()
        if cached is not None and not cached.empty:
            hist = cached[cached["Date"] < s_start].copy()
            if not hist.empty:
                common_cols = df_sim.columns.intersection(hist.columns)
                df_sim = pd.concat([hist[common_cols], df_sim[common_cols]],
                                   ignore_index=True)
                df_sim = df_sim.sort_values("Date").reset_index(drop=True)
        config0 = BacktestConfig(**BASE_KWARGS)
        df_sim = compute_indicators(
            df_sim, entry_period=config0.entry_period, exit_period=config0.exit_period,
            atr_period=config0.atr_period, ema_fast=config0.ema_fast,
            ema_slow=config0.ema_slow, channel_type=config0.channel_type,
            bb_period=config0.bb_period, bb_k=config0.bb_k,
            am_ma_period=config0.am_ma_period, mode=config0.mode,
        )
        sim_dfs[label] = df_sim
        sim_mask = df_sim["Date"] >= s_start
        sc = df_sim.loc[sim_mask, "Close"]
        sim_benchmarks[label] = (sc.iloc[-1] / sc.iloc[0] - 1) * 100 if len(sc) >= 2 else 0.0
        print(f"     ✓ {label}: ¥{sc.iloc[0]:.1f}→¥{sc.iloc[-1]:.1f} ({sim_benchmarks[label]:+.1f}%)")

    # 收集所有候选×所有场景的 stats，统一百分位归一化
    all_sim_stats = []
    for label, params in candidates:
        sim_kwargs = {**BASE_KWARGS, "pm_start_date": sim_start}
        for sc_label in SCENARIO_LABEL_MAP:
            stats_s = run_one_backtest(params, sim_kwargs, sim_dfs[sc_label],
                                       benchmark_return=sim_benchmarks[sc_label])
            if stats_s:
                stats_s["_cand_label"] = label
                stats_s["_sc_label"] = sc_label
                all_sim_stats.append(stats_s)

    # 统一百分位归一化
    sim_weighted = {}
    if len(all_sim_stats) >= 3:
        compute_percentile_scores(all_sim_stats)
        for label, _ in candidates:
            cand_stats = [s for s in all_sim_stats if s["_cand_label"] == label]
            if cand_stats:
                weighted = 0.0
                total_prob = 0.0
                for s in cand_stats:
                    prob = SCENARIO_PROBS.get(s["_sc_label"], 0.0)
                    weighted += s["score"] * prob
                    total_prob += prob
                sim_weighted[label] = round(weighted / total_prob, 1) if total_prob > 0 else 0.0
            else:
                sim_weighted[label] = 0.0
    else:
        for label, _ in candidates:
            sim_weighted[label] = 50.0

    # 打印模拟得分表
    print(f"\n  {'候选':<14} ", end="")
    for sc_label in SCENARIO_LABEL_MAP:
        print(f"{sc_label[:2]:>6}", end="")
    print(f" {'加权':>7}")
    print(f"  {'─' * 70}")
    for label, params in candidates:
        print(f"  {label:<14} ", end="")
        for sc_label in SCENARIO_LABEL_MAP:
            matches = [s for s in all_sim_stats
                       if s["_cand_label"] == label and s["_sc_label"] == sc_label]
            if matches:
                print(f"{matches[0]['score']:>6.1f}", end="")
            else:
                print(f"{'N/A':>6}", end="")
        print(f" {sim_weighted.get(label, 0):>7.1f}")

    # ── 汇总排名 ──
    # 综合得分 = Part1全量得分 × 1/3 + 滚动均值(百分位) × 1/3 + 模拟加权(百分位) × 1/3
    # P10 仅作稳定性参考，不参与公式
    print(f"\n  ═══ 综合排名 ═══")
    print(f"  {'候选':<14} {'Part1得分':>8} {'滚动均分':>8} {'模拟加权':>8} {'综合':>8} {'P10':>8} {'稳定性'}")
    print(f"  {'─' * 72}")
    final = []
    for label, params in candidates:
        p1 = part1_scores.get(label, 50.0)
        r_mean = rolling_scores.get(label, 0)
        sw = sim_weighted.get(label, 50.0)
        combined = p1 * (1/3.0) + r_mean * (1/3.0) + sw * (1/3.0)
        sc = candidate_window_scores.get(label, [])
        p10 = np.percentile(sc, 10) if len(sc) >= 10 else 0.0
        stable = "✅" if p10 > p10_threshold else "⚠️"
        final.append((label, params, p1, r_mean, sw, combined, p10))
        print(f"  {label:<14} {p1:>8.1f} {r_mean:>8.1f} {sw:>8.1f} {combined:>8.1f} {p10:>8.1f} {stable}")

    final.sort(key=lambda x: x[5], reverse=True)
    print(f"\n  🏆 最优: {final[0][0]} (综合={final[0][5]:.1f})")
    print(f"     gd={final[0][1]['pm_grid_down']:.2f} gu={final[0][1]['pm_grid_up']:.2f} "
          f"rp={final[0][1]['pm_reduce_pct']:.2f} ap={final[0][1]['pm_add_pct']:.2f} "
          f"k={final[0][1]['pm_agg_k']:.2f} adx={final[0][1]['pm_adx_threshold']:.2f}")

    print()
    return {"candidate_scores": candidate_window_scores, "sim_weighted": sim_weighted,
            "part1_scores": part1_scores, "final": final, "window_details": window_details}



# ═══════════════════════════════════════════════════════════════
# Part 1: 全量搜索
# ═══════════════════════════════════════════════════════════════

def part1_global_search(df_base: pd.DataFrame):
    """Part 1: 全量搜索 —— 完整历史上按综合得分排名，找到离散全局最优。"""
    print("=" * 70)
    print("🐢 Part 1: 全量搜索 —— 综合得分排名")
    print("=" * 70)

    bm = compute_buy_hold(df_base, BASE_KWARGS["pm_start_date"])

    gd_vals = SWEEP_RANGES["pm_grid_down"]
    gu_vals = SWEEP_RANGES["pm_grid_up"]
    rp_vals = SWEEP_RANGES["pm_reduce_pct"]
    ap_vals = SWEEP_RANGES["pm_add_pct"]
    k_vals  = SWEEP_RANGES["pm_agg_k"]

    batch = []
    t0 = time.time()
    total = len(gd_vals) * len(gu_vals) * len(rp_vals) * len(ap_vals) * 1 * len(k_vals) * 1

    for gd, gu, rp, ap, adx, k, tp in itertools.product(
            gd_vals, gu_vals, rp_vals, ap_vals, [25], k_vals, [1.30]):
        params = dict(pm_grid_down=gd, pm_grid_up=gu, pm_reduce_pct=rp,
                     pm_add_pct=ap, pm_adx_threshold=adx, pm_agg_k=k,
                     pm_take_profit_ratio=tp)
        stats = run_one_backtest(params, BASE_KWARGS, df_base, benchmark_return=bm)
        if stats:
            stats["_params"] = params
            batch.append(stats)

    # 批量百分位归一化
    compute_percentile_scores(batch)

    all_stats = []
    for s in batch:
        p = s["_params"]
        all_stats.append({
            **p,
            "score": s["score"],
            "CAGR": round(s["CAGR (%)"], 2),
            "MaxDD": round(s["最大回撤 (%)"], 2),
            "excess": round(s["超额收益 (%)"], 2),
            "robust_sharpe": round(s["稳健夏普比率"], 2),
            "r_cubed": round(s["R立方"], 2),
            "trades": s["总交易"],
        })

    all_stats.sort(key=lambda x: x["score"], reverse=True)

    header = f"  {'#':<3} {'百分位':>8} {'CAGR%':>7} {'DD%':>7} {'超额%':>7} {'笔':>4}  gd   gu   rp   ap    k"
    print(f"\n{header}")
    print(f"  {'─' * 70}")
    for i, r in enumerate(all_stats[:10], 1):
        print(f"  {i:<3} {r['score']:>8.2f} {r['CAGR']:>7.2f} {r['MaxDD']:>7.2f} "
              f"{r['excess']:>7.2f} {r['trades']:>4}  "
              f"{r['pm_grid_down']:.2f} {r['pm_grid_up']:.2f} "
              f"{r['pm_reduce_pct']:.2f} {r['pm_add_pct']:.2f} {r['pm_agg_k']:.2f}")

    print(f"\n  ... 最差百分位得分(底部5名):")
    for i, r in enumerate(all_stats[-5:], len(all_stats) - 4):
        print(f"  {i:<3} {r['score']:>8.2f} {r['CAGR']:>7.2f} {r['MaxDD']:>7.2f} "
              f"{r['excess']:>7.2f} {r['trades']:>4}  "
              f"{r['pm_grid_down']:.2f} {r['pm_grid_up']:.2f} "
              f"{r['pm_reduce_pct']:.2f} {r['pm_add_pct']:.2f} {r['pm_agg_k']:.2f}")

    top10 = all_stats[:10]
    print(f"\n  TOP10 聚集: gd∈{sorted(set(r['pm_grid_down'] for r in top10))}  "
          f"gu∈{sorted(set(r['pm_grid_up'] for r in top10))}  "
          f"rp∈{sorted(set(r['pm_reduce_pct'] for r in top10))}  "
          f"k∈{sorted(set(r['pm_agg_k'] for r in top10))}")

    print(f"\n  耗时: {time.time()-t0:.0f}s  有效: {len(all_stats)}/{total}")
    return all_stats


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    print("grid_search_pm.py已废弃")
    # print("╔══════════════════════════════════════════════════════════╗")
    # print("║  海龟交易法则 第十一章 最优化实验                        ║")
    # print("║  参数搜索空间与 grid_search_pm.py 完全一致              ║")
    # print("╚══════════════════════════════════════════════════════════╝\n")
    #
    # print("📥 加载数据 ...")
    # config0 = BacktestConfig(**BASE_KWARGS)
    # df_base = fetch_data(
    #     ts_code=config0.ts_code, stock_name=config0.stock_name,
    #     market=config0.market, start_date=config0.start_date,
    #     end_date=config0.end_date, exchange=config0.exchange,
    #     data_source=config0.data_source,
    # )
    # df_base = compute_indicators(
    #     df_base, entry_period=config0.entry_period,
    #     exit_period=config0.exit_period, atr_period=config0.atr_period,
    #     ema_fast=config0.ema_fast, ema_slow=config0.ema_slow,
    #     channel_type=config0.channel_type, bb_period=config0.bb_period,
    #     bb_k=config0.bb_k, am_ma_period=config0.am_ma_period,
    #     mode=config0.mode,
    # )
    # print(f"   数据: {len(df_base)} 条\n")
    #
    # # Part 1: 全量搜索
    # all_stats = part1_global_search(df_base)
    #
    # # 用全局最优作为 Part 2 的起点
    # if all_stats:
    #     global_best = all_stats[0]
    #     DEFAULT_PARAMS.update({
    #         "pm_grid_down": global_best["pm_grid_down"],
    #         "pm_grid_up": global_best["pm_grid_up"],
    #         "pm_reduce_pct": global_best["pm_reduce_pct"],
    #         "pm_add_pct": global_best["pm_add_pct"],
    #         "pm_agg_k": global_best["pm_agg_k"],
    #     })
    #     print(f"  📌 已用全局最优更新 DEFAULT_PARAMS: "
    #           f"gd={DEFAULT_PARAMS['pm_grid_down']:.2f} "
    #           f"gu={DEFAULT_PARAMS['pm_grid_up']:.2f} "
    #           f"rp={DEFAULT_PARAMS['pm_reduce_pct']:.2f} "
    #           f"ap={DEFAULT_PARAMS['pm_add_pct']:.2f} "
    #           f"k={DEFAULT_PARAMS['pm_agg_k']:.2f}\n")
    #
    # # Part 2: 单参数扫描
    # sweeps, greedy_params, second_best = part2_param_sweep(df_base)
    #
    # # Part 3: 滚动窗口 + 模拟数据
    # part3_results = part3_rolling_sim(df_base, greedy_params, second_best,
    #                                    part1_all_stats=all_stats)
    #
    # # 保存
    # out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    #     os.path.abspath(__file__)))), "result")
    # os.makedirs(out_dir, exist_ok=True)
    #
    # if sweeps:
    #     _plot_sweep_curves(sweeps, out_dir,
    #                        filename="turtle_sweeps_part2.png",
    #                        title="Part 2 收敛后 — 单参数扫描",
    #                        base_params=sweeps.get("_base_params"))
    #
    # if part3_results:
    #     xlsx_path = os.path.join(out_dir, "turtle_rolling_sim.xlsx")
    #     with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
    #         if part3_results.get("window_details"):
    #             pd.DataFrame(part3_results["window_details"]).to_excel(
    #                 writer, sheet_name="每窗口明细", index=False)
    #         if part3_results.get("final"):
    #             pd.DataFrame(part3_results["final"],
    #                          columns=["label", "params", "part1_score", "rolling_mean",
    #                                  "sim_weighted", "combined", "p10"]).to_excel(
    #                 writer, sheet_name="综合排名", index=False)
    #
    #     # ── Part 3 验证后：对综合排名 top 3 分别做单参数扫描 ──
    #     if part3_results.get("final"):
    #         bm = compute_buy_hold(df_base, BASE_KWARGS["pm_start_date"])
    #         top3 = part3_results["final"][:3]
    #         for rank, (label, params, p1, r_mean, sw, combined, p10) in enumerate(top3, 1):
    #             print(f"\n  🔬 Top{rank} 单参数扫描: {label} "
    #                   f"(gd={params['pm_grid_down']:.3f} gu={params['pm_grid_up']:.3f} "
    #                   f"rp={params['pm_reduce_pct']:.3f} ap={params['pm_add_pct']:.3f} "
    #                   f"k={params['pm_agg_k']:.3f})")
    #             sweeps = _single_round_sweep(df_base, params, bm)
    #             if sweeps:
    #                 safe_label = label.replace("/", "_").replace(" ", "_")
    #                 _plot_sweep_curves(sweeps, out_dir,
    #                                    filename=f"turtle_sweeps_top{rank}_{safe_label}.png",
    #                                    title=f"Top{rank} 单参数扫描 — {label} (综合={combined:.1f})",
    #                                    base_params=sweeps.get("_base_params"))
    #
    # if all_stats:
    #     pd.DataFrame(all_stats).to_excel(
    #         os.path.join(out_dir, "turtle_full_search.xlsx"), index=False)
    #
    # print(f"\n📁 结果已保存: result/turtle_*.xlsx")
    # print("✅ 海龟第十一章实验完成！")


if __name__ == "__main__":
    main()
