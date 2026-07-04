#!/usr/bin/env python3
"""海龟策略单参数扫描 — 基于最优参数，逐参数扫描 + 三类窗口综合评分 + 绘图

用法：
  python strategy/position_management_1/turtle_sweep.py
"""

import copy
import os
import sys
import time
from datetime import timedelta

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from core.config import BacktestConfig
from core.data import fetch_data
from core.indicators import compute_indicators
from core.scoring import score_pipeline

# ── 复用 ──
from strategy.position_management_1.turtle_optimize import (
    run_one_backtest, _constraint_ok,
    _plot_sweep_curves, _minmax_norm, PARAM_LABELS,
)
from strategy.position_management_1.grid_search_pm import (
    prepare_scenario_dfs, SCENARIO_LABEL_MAP, SCENARIO_PROBS,
)
from strategy.position_management_1.simulate_market_scenarios import (
    get_last_price as sim_get_last_price,
)

# ═══════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════

BASE_KWARGS = dict(
    ts_code="09992.HK", stock_name="泡泡玛特", market="HK",
    mode="position_management_1", per_shares=200,
    start_date="20201211", pm_start_date="20210112",
    user_dmi=True, pm_adaptive=True,
    pm_adaptive_grid=False, pm_dynamic_position=False,
    ema_fast=0, ema_slow=0, pm_initial_shares=2,
    initial_capital=40000, pm_agg_k=1.0, pm_take_profit_ratio=1.30,
    data_source="tushare",
)

# 最优参数（来自 turtle_scoring.py 综合排名第一）
OPTIMAL_PARAMS = dict(
    pm_grid_down=0.05, pm_grid_up=0.028, pm_reduce_pct=0.50,
    pm_add_pct=0.40, pm_adx_threshold=35,
    pm_agg_k=1.0,  # 屏蔽（_plot_sweep_curves 需要此key）
)

# 固定熵权（来自 turtle_scoring.py 输出，不再每次重新计算）
# 运行 turtle_scoring.py 后，将打印的权重填入此处
SCORING_WEIGHTS = dict(
    excess=0.50, robust_sharpe=0.40, r_cubed=0.10,
)

# 单参数扫描定义（围绕最优值展开）
SWEEP_DEFS = {
    "pm_grid_down": {
        "values": [0.04,0.045, 0.05,0.055, 0.06,0.065,0.07,0.075, 0.08,0.085,0.09,0.095,0.10,0.105,0.11,0.115,0.12],
        "label": PARAM_LABELS.get("pm_grid_down", "pm_grid_down"),
    },
    "pm_grid_up": {
        "values": [0.025,0.026,0.027,0.028,0.029,0.03,0.031,0.032,0.033,0.034, 0.035,0.036,0.037,0.038,0.039, 0.04,0.041,0.042,0.043,0.044,0.045,0.046,0.047,0.048,0.049, 0.05,0.055, 0.06,0.065,0.07,0.075, 0.08,0.085,0.09,0.095,0.10,0.105,0.11],
        "label": PARAM_LABELS.get("pm_grid_up", "pm_grid_up"),
    },
    "pm_reduce_pct": {
        "values": [0.45,0.46,0.47,0.48,0.49,0.50,0.51,0.52,0.53,0.54,0.55,0.56,0.57,0.58,0.59, 0.60,0.61,0.62,0.63,0.64,0.65,0.66,0.67,0.68,0.69,0.70,0.71,0.72,0.73,0.74,0.75],
        "label": PARAM_LABELS.get("pm_reduce_pct", "pm_reduce_pct"),
    },
    "pm_add_pct": {
        "values": [0.20,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29,0.30,0.31,0.32,0.33,0.34,0.35,0.36,0.37,0.38,0.39,0.40,0.41,0.42,0.43,0.44,0.45,0.50,0.55],
        "label": PARAM_LABELS.get("pm_add_pct", "pm_add_pct"),
    },
    "pm_adx_threshold": {
        "values": [20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45],
        "label": PARAM_LABELS.get("pm_adx_threshold", "pm_adx_threshold"),
    },
}

# 滚动窗口参数
WINDOW_BARS = 252
STEP_BARS = 20
WARMUP = 60

# ═══════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════

def _now():
    return time.strftime("%H:%M:%S")


def _compute_buy_hold(df: pd.DataFrame, start_date: str) -> float:
    mask = df["Date"] >= start_date
    close = df.loc[mask, "Close"]
    if len(close) < 2:
        return 0.0
    return round((close.iloc[-1] / close.iloc[0] - 1) * 100, 2)


def _run_three_windows(param_combos: list[dict], base_kwargs: dict,
                       df_base: pd.DataFrame, scenario_dfs: dict,
                       scenario_benchmarks: dict, sim_start: str,
                       full_bm: float) -> list[dict]:
    """对一组参数组合运行三类窗口回测，返回原始结果列表。"""
    all_results = []
    dates = pd.to_datetime(df_base["Date"])
    n = len(df_base)
    n_windows = max(0, (n - WARMUP - WINDOW_BARS) // STEP_BARS + 1)

    # Stage A: 全量历史
    for params in param_combos:
        stats = run_one_backtest(params, base_kwargs, df_base, benchmark_return=full_bm)
        if stats:
            stats["_window"] = "full"
            stats["_params"] = params
            all_results.append(stats)

    # Stage B: 滚动窗口
    for params in param_combos:
        for start_idx in range(WARMUP, n - WINDOW_BARS, STEP_BARS):
            end_idx = start_idx + WINDOW_BARS
            warmup_start = max(0, start_idx - WARMUP)
            df_win = df_base.iloc[warmup_start:end_idx].copy()
            ws = pd.Timestamp(dates.iloc[start_idx]).strftime("%Y%m%d")

            win_mask = df_win["Date"] >= ws
            win_close = df_win.loc[win_mask, "Close"]
            bm = ((win_close.iloc[-1] / win_close.iloc[0] - 1) * 100
                  if len(win_close) >= 2 else 0.0)

            win_kwargs = {**base_kwargs, "pm_start_date": ws}
            stats = run_one_backtest(params, win_kwargs, df_win, benchmark_return=bm)
            if stats and stats.get("总交易", 0) > 0:
                stats["_window"] = f"rolling_{ws}"
                stats["_params"] = params
                all_results.append(stats)

    # Stage C: 8场景模拟
    for params in param_combos:
        for sc_label in SCENARIO_LABEL_MAP:
            sc_kwargs = {**base_kwargs, "pm_start_date": sim_start}
            stats = run_one_backtest(params, sc_kwargs,
                                     scenario_dfs[sc_label],
                                     benchmark_return=scenario_benchmarks[sc_label])
            if stats and stats.get("总交易", 0) > 0:
                stats["_window"] = f"sim_{sc_label}"
                stats["_params"] = params
                all_results.append(stats)

    return all_results


def _aggregate_sweep(all_results: list[dict], sweep_param: str) -> list[dict]:
    """按扫描参数值汇总 → 每个值一个综合得分。"""
    from collections import defaultdict
    param_groups = defaultdict(list)
    for r in all_results:
        val = r["_params"][sweep_param]
        param_groups[val].append(r)

    summary = []
    for val, group in param_groups.items():
        params = group[0]["_params"]

        full_scores = [r["score"] for r in group if r["_window"] == "full"]
        full_score = full_scores[0] if full_scores else 50.0

        rolling_scores = [r["score"] for r in group
                          if r["_window"].startswith("rolling_")]
        if len(rolling_scores) >= 5:
            rolling_median = float(np.median(rolling_scores))
            rolling_p10 = float(np.percentile(rolling_scores, 10))
        else:
            rolling_median = float(np.mean(rolling_scores)) if rolling_scores else 50.0
            rolling_p10 = rolling_median

        sim_scores, sim_probs = [], []
        for r in group:
            if r["_window"].startswith("sim_"):
                sc_label = r["_window"].replace("sim_", "")
                prob = SCENARIO_PROBS.get(sc_label, 0.0)
                if prob > 0:
                    sim_scores.append(r["score"] * prob)
                    sim_probs.append(prob)

        sim_weighted = (sum(sim_scores) / sum(sim_probs)
                        if sim_scores and sum(sim_probs) > 0 else 50.0)

        combined = 0.5 * rolling_median + 0.3 * sim_weighted + 0.2 * full_score

        full_orig = next((r for r in group if r["_window"] == "full"), group[0])
        summary.append({
            "value": val,
            "score": round(combined, 2),
            "excess": round(full_orig.get("excess",
                             full_orig.get("超额收益 (%)", 0) or 0), 2),
            "robust_sharpe": round(full_orig.get("robust_sharpe",
                                   full_orig.get("稳健夏普比率", 0) or 0), 2),
            "r_cubed": round(full_orig.get("r_cubed",
                             full_orig.get("R立方", 0) or 0), 2),
            "CAGR": round(full_orig.get("CAGR (%)", 0), 2),
            "MaxDD": round(full_orig.get("最大回撤 (%)", 0), 2),
            "trades": full_orig.get("总交易", 0),
            "rolling_median": round(rolling_median, 2),
            "rolling_p10": round(rolling_p10, 2),
            "sim_weighted": round(sim_weighted, 2),
            "full_score": round(full_score, 2),
        })

    summary.sort(key=lambda x: x["value"])
    return summary


def _build_sweep_dict(summary: list[dict], sweep_param: str, base_params: dict):
    """将汇总结果转为 _plot_sweep_curves 需要的 sweeps 格式。"""
    results = []
    for s in summary:
        results.append({
            "value": s["value"],
            "score": s["score"],
            "CAGR": s["CAGR"],
            "MaxDD": s["MaxDD"],
            "excess": s["excess"],
            "robust_sharpe": s["robust_sharpe"],
            "r_cubed": s["r_cubed"],
            "trades": s["trades"],
        })

    sweeps = {sweep_param: results, "_base_params": base_params}
    return sweeps


# ═══════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  海龟策略单参数扫描 — 三类窗口综合评分                    ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"\n  基准参数: gd={OPTIMAL_PARAMS['pm_grid_down']:.2f}  "
          f"gu={OPTIMAL_PARAMS['pm_grid_up']:.2f}  "
          f"rp={OPTIMAL_PARAMS['pm_reduce_pct']:.2f}  "
          f"ap={OPTIMAL_PARAMS['pm_add_pct']:.2f}  "
          f"adx={int(OPTIMAL_PARAMS['pm_adx_threshold'])}")

    # ── 加载数据 ──
    print(f"\n[{_now()}] 📥 加载数据 ...")
    config0 = BacktestConfig(**BASE_KWARGS)
    df_base = fetch_data(
        ts_code=config0.ts_code, stock_name=config0.stock_name,
        market=config0.market, start_date=config0.start_date,
        end_date=config0.end_date, exchange=config0.exchange,
        data_source=config0.data_source,
    )
    df_base = compute_indicators(
        df_base, entry_period=config0.entry_period, exit_period=config0.exit_period,
        atr_period=config0.atr_period, ema_fast=config0.ema_fast,
        ema_slow=config0.ema_slow, channel_type=config0.channel_type,
        bb_period=config0.bb_period, bb_k=config0.bb_k,
        am_ma_period=config0.am_ma_period, mode=config0.mode,
    )
    full_bm = _compute_buy_hold(df_base, BASE_KWARGS["pm_start_date"])
    print(f"   数据: {len(df_base)} 条, 买入持有基准: {full_bm:+.2f}%")

    # ── 准备场景数据（只需一次）──
    print(f"\n[{_now()}] 📊 准备8种模拟场景 ...")
    try:
        last_price, last_date = sim_get_last_price()
    except Exception:
        from core.cache import load as cache_load
        cached = cache_load(config0.ts_code, config0.market)
        last_price = float(cached["Close"].iloc[-1])
        last_date = cached["Date"].iloc[-1]
    sim_start = (last_date + timedelta(days=1)).strftime("%Y%m%d")

    scenario_dfs, scenario_benchmarks = prepare_scenario_dfs(
        last_price, last_date, config0, df_base)
    print(f"   模拟起始: {sim_start}, 最新价: {last_price:.2f}")

    # ── 输出目录 ──
    out_dir = os.path.join(PROJECT_ROOT, "result")
    os.makedirs(out_dir, exist_ok=True)

    # ── 逐参数扫描（迭代收敛）──
    round_num = 0
    while True:
        round_num += 1
        print(f"\n{'🔄'*30}")
        print(f"[{_now()}] 🔄 第 {round_num} 轮扫描")
        print(f"  基准参数: gd={OPTIMAL_PARAMS['pm_grid_down']:.3f}  "
              f"gu={OPTIMAL_PARAMS['pm_grid_up']:.3f}  "
              f"rp={OPTIMAL_PARAMS['pm_reduce_pct']:.3f}  "
              f"ap={OPTIMAL_PARAMS['pm_add_pct']:.3f}  "
              f"adx={int(OPTIMAL_PARAMS['pm_adx_threshold'])}")
        print(f"{'🔄'*30}")

        param_changed = False
        all_sweeps = {}
        all_summaries = {}

        for pname, pdef in SWEEP_DEFS.items():
            sweep_vals = pdef["values"]
            label = pdef["label"]

            print(f"\n{'='*60}")
            print(f"[{_now()}] 🔍 扫描 {label} ({pname}): {len(sweep_vals)} 个值")
            print(f"{'='*60}")

            # 生成参数组合（使用当前 OPTIMAL_PARAMS）
            param_combos = []
            skipped = []
            for val in sweep_vals:
                params = copy.deepcopy(OPTIMAL_PARAMS)
                params[pname] = round(val, 3)
                if _constraint_ok(params):
                    param_combos.append(params)
                else:
                    skipped.append(val)
            if skipped:
                print(f"  ⛔ 约束跳过: {skipped}")
            print(f"  有效组合: {len(param_combos)}")

            # 三窗口回测
            t0 = time.time()
            raw_results = _run_three_windows(
                param_combos, BASE_KWARGS, df_base,
                scenario_dfs, scenario_benchmarks, sim_start, full_bm)
            print(f"  回测完成: {len(raw_results)} 条, 耗时 {(time.time()-t0)/60:.1f}min")

            # 评分管线
            t0 = time.time()
            scored_results, weights = score_pipeline(raw_results, fixed_weights=SCORING_WEIGHTS)
            print(f"  评分完成: 固定权重 { {k:round(v,3) for k,v in weights.items()} }")

            # 汇总
            summary = _aggregate_sweep(scored_results, pname)
            for s in summary:
                print(f"    {label}={s['value']:.3f} → 综合={s['score']:.1f}  "
                      f"超额={s['excess']:+.1f}% 稳夏={s['robust_sharpe']:.2f}  "
                      f"R³={s['r_cubed']:.2f} 回撤={s['MaxDD']:.1f}%")

            # 收集 sweeps 用于最终绘图
            sweeps = _build_sweep_dict(summary, pname, OPTIMAL_PARAMS)
            all_sweeps[pname] = sweeps[pname]
            all_summaries[pname] = summary

            # 检查最优值是否改变（值变化 + 评分提升 > 0.1 才算改变）
            best = max(summary, key=lambda x: x["score"])
            best_val = best["value"]
            best_score = best["score"]
            old_val = OPTIMAL_PARAMS[pname]

            # 找到当前参数值对应的评分
            old_entry = next((s for s in summary if abs(s["value"] - old_val) < 0.0005), None)
            old_score = old_entry["score"] if old_entry else best_score

            if abs(best_val - old_val) > 0.001 and (best_score - old_score) > 0.1:
                print(f"  🔄 {pname} 最优值改变: {old_val:.3f}(得分{old_score:.1f}) → "
                      f"{best_val:.3f}(得分{best_score:.1f})")
                OPTIMAL_PARAMS[pname] = round(best_val, 3)
                param_changed = True
                break  # 停止本轮扫描，用新参数重新开始
            else:
                print(f"  ✅ {pname} 最优值未变: {best_val:.3f} (得分{best_score:.1f})")

        # ── 收敛判断 ──
        if not param_changed:
            # 完整扫描一轮且无任何参数改变 → 收敛
            all_sweeps["_base_params"] = OPTIMAL_PARAMS
            print(f"\n[{_now()}] ✅ 第 {round_num} 轮收敛！最优参数不再改变。")
            break
        else:
            print(f"\n[{_now()}] ⚠️ 最优参数已改变，重新扫描全部参数...")

    # ── 最终绘图：5张图合并为一个文件 ──
    _plot_sweep_curves(
        all_sweeps, out_dir,
        filename="turtle_sweep_all.png",
        title=f"单参数扫描 — 收敛后 (gd={OPTIMAL_PARAMS['pm_grid_down']:.3f} "
              f"gu={OPTIMAL_PARAMS['pm_grid_up']:.3f} "
              f"rp={OPTIMAL_PARAMS['pm_reduce_pct']:.3f} "
              f"ap={OPTIMAL_PARAMS['pm_add_pct']:.3f} "
              f"adx={int(OPTIMAL_PARAMS['pm_adx_threshold'])})",
        base_params=OPTIMAL_PARAMS,
    )

    # ── 保存 Excel（所有参数汇总到一个文件）──
    with pd.ExcelWriter(os.path.join(out_dir, "turtle_sweep_all.xlsx"),
                        engine="openpyxl") as writer:
        for pname, summary in all_summaries.items():
            label = SWEEP_DEFS[pname]["label"]
            sheet_name = pname.replace("pm_", "")[:31]  # Excel sheet name ≤ 31 chars
            pd.DataFrame(summary).to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"\n[{_now()}] ✅ 全部扫描完成！")
    print(f"  最终参数: gd={OPTIMAL_PARAMS['pm_grid_down']:.3f}  "
          f"gu={OPTIMAL_PARAMS['pm_grid_up']:.3f}  "
          f"rp={OPTIMAL_PARAMS['pm_reduce_pct']:.3f}  "
          f"ap={OPTIMAL_PARAMS['pm_add_pct']:.3f}  "
          f"adx={int(OPTIMAL_PARAMS['pm_adx_threshold'])}")
    print(f"  图表: result/turtle_sweep_all.png")
    print(f"  Excel: result/turtle_sweep_all.xlsx")


if __name__ == "__main__":
    main()
