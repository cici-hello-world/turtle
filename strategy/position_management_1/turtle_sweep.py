#!/usr/bin/env python3
"""海龟策略单参数扫描 — 基于最优参数，逐参数扫描 + 三类窗口综合评分 + 绘图

用法：
  python strategy/position_management_1/turtle_sweep.py
"""

import copy
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import timedelta
import multiprocessing

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from core.config import BacktestConfig
from core.data import fetch_data
from core.indicators import compute_indicators
from core.scoring import score_pipeline, AbsoluteScorer

# ── 复用 ──
from strategy.position_management_1.turtle_optimize import (
    run_one_backtest, _constraint_ok,
    _plot_sweep_curves, _minmax_norm, PARAM_LABELS,
)
from strategy.position_management_1.grid_search_pm import (
    prepare_scenario_dfs, SCENARIO_LABEL_MAP, get_scenario_probs,
)
from strategy.position_management_1.simulate_market_scenarios import (
    get_last_price as sim_get_last_price,
)

# ═══════════════════════════════════════════════════════════════
# 配置 — key 为 ts_code，通过 init_config(ts_code) 激活
# ═══════════════════════════════════════════════════════════════

_ALL_BASE_KWARGS = {
    "09992.HK": dict(  # 泡泡玛特
        ts_code="09992.HK", stock_name="泡泡玛特", market="HK",
        mode="position_management_1", per_shares=200,
        start_date="20201211", pm_start_date="20210112",
        user_dmi=True, pm_adaptive=True, pm_adaptive_grid=False,
        pm_dynamic_position=False, ema_fast=0, ema_slow=0,
        pm_initial_shares=2, initial_capital=40000, pm_agg_k=1.0,
        data_source="tushare",
    ),
    "03690.HK": dict(  # 美团
        ts_code="03690.HK", stock_name="美团", market="HK",
        mode="position_management_1", per_shares=100,
        pm_start_date="20181023", start_date="20180920",
        user_dmi=True, pm_adaptive=True, pm_adaptive_grid=False,
        pm_dynamic_position=False, ema_fast=0, ema_slow=0,
        pm_initial_shares=2, initial_capital=40000,
        data_source="tushare",
    ),
    "02015.HK": dict(  # 理想
        ts_code="02015.HK", stock_name="理想", market="HK",
        mode="position_management_1", per_shares=100,
        pm_start_date="20210909", start_date="20210812",
        user_dmi=True, pm_adaptive=False, pm_adaptive_grid=False,
        pm_dynamic_position=False, ema_fast=0, ema_slow=0,
        pm_initial_shares=25, initial_capital=40000,
        data_source="tushare",
    ),
    "06181.HK": dict(  # 老铺
            ts_code="06181.HK",stock_name="老铺",market="HK",
            mode="position_management_1",per_shares=100,
            pm_start_date="20240729",start_date="20240628",
            user_dmi=True,pm_adaptive=True,pm_adaptive_grid=True,
            pm_dynamic_position=False,ema_fast=0,ema_slow=0,
            pm_initial_shares=3,initial_capital=40000,
            data_source="tushare",
        ),
    "01810.HK": dict(  # 小米
        ts_code="01810.HK", stock_name="小米", market="HK",
        mode="position_management_1",per_shares=200,
        pm_start_date="20180806", start_date="20180709",
        user_dmi=True,pm_adaptive=True,pm_adaptive_grid=False,
        pm_dynamic_position=False,ema_fast=0,ema_slow=0,
        pm_initial_shares=4, initial_capital=40000,
        data_source="tushare",
    ),
    "000625.SZ": dict(  # 长安
        ts_code="000625.SZ", stock_name="长安", market="A",
        mode="position_management_1",per_shares=100,
        pm_start_date="20160729", start_date="20160701",
        user_dmi=True,pm_adaptive=True,pm_adaptive_grid=False,
        pm_dynamic_position=False,ema_fast=0,ema_slow=0,
        pm_initial_shares=9, initial_capital=40000,
        data_source="tushare",
    ),
}

_ALL_OPTIMAL_PARAMS = {
    "09992.HK": dict(  # 泡泡玛特
        pm_grid_down=0.15,pm_grid_up=0.038,pm_reduce_pct=0.79,
        pm_add_pct=0.50,pm_adx_threshold=21,pm_agg_k=1,
    ),
    "03690.HK": dict(  # 美团
        pm_grid_down=0.105, pm_grid_up=0.039, pm_reduce_pct=0.52,
        pm_add_pct=0.5, pm_adx_threshold=35, pm_agg_k=1,
    ),
    "02015.HK": dict(  # 理想
        pm_grid_down=0.12, pm_grid_up=0.027, pm_reduce_pct=0.51,
        pm_add_pct=0.48, pm_adx_threshold=36, pm_agg_k=1,
    ),
    "06181.HK": dict(  # 老铺
        pm_grid_down=0.075, pm_grid_up=0.05, pm_reduce_pct=0.75,
        pm_add_pct=0.50, pm_adx_threshold=36, pm_agg_k=0.9,
    ),
    "01810.HK": dict(  # 小米
        pm_grid_down=0.07, pm_grid_up=0.05, pm_reduce_pct=0.56,
        pm_add_pct=0.33, pm_adx_threshold=27, pm_agg_k=1,
    ),
    "000625.SZ": dict(  # 长安
        pm_grid_down=0.08, pm_grid_up=0.028, pm_reduce_pct=0.66,
        pm_add_pct=0.33, pm_adx_threshold=20, pm_agg_k=1,
    ),
}

# 运行时激活的配置（由 init_config 设置）
BASE_KWARGS = None
OPTIMAL_PARAMS = None


def init_config(ts_code: str):
    """根据 ts_code 激活对应的 BASE_KWARGS 和 OPTIMAL_PARAMS。"""
    global BASE_KWARGS, OPTIMAL_PARAMS
    if ts_code not in _ALL_BASE_KWARGS:
        raise KeyError(f"未配置 {ts_code} 的 BASE_KWARGS，请在 _ALL_BASE_KWARGS 中添加")
    if ts_code not in _ALL_OPTIMAL_PARAMS:
        raise KeyError(f"未配置 {ts_code} 的 OPTIMAL_PARAMS，请在 _ALL_OPTIMAL_PARAMS 中添加")
    BASE_KWARGS = _ALL_BASE_KWARGS[ts_code]
    OPTIMAL_PARAMS = _ALL_OPTIMAL_PARAMS[ts_code]

# 绝对评分配置 — key 为 ts_code，通过 get_scorer(ts_code) 获取
SCORING_CONFIG = {
    "02015.HK": dict(  # 理想
        weights=dict(excess=0.387,  robust_sharpe=0.321,  r_cubed=0.293),
        scoring_params=dict(
            excess=dict(type="linear_quantile", slope=1.89, intercept=-78.73),
            robust_sharpe=dict(type="linear_quantile", slope=183.67, intercept=25.20),
            r_cubed=dict(type="linear_quantile", slope=11.04, intercept=6.66),
        ),
    ),
    "09992.HK": dict(  # 泡泡玛特
        weights=dict(excess=0.376,  robust_sharpe=0.340,  r_cubed=0.285),
        scoring_params=dict(
            excess=dict(type="linear_quantile", slope=0.22, intercept=39.17),
            robust_sharpe=dict(type="linear_quantile", slope=63.83, intercept=11.38),
            r_cubed=dict(type="linear_quantile", slope=19.52, intercept=8.32),
        ),
    ),
    "03690.HK": dict(  # 美团
        weights=dict(excess=0.332,  robust_sharpe=0.407,  r_cubed=0.261),
        scoring_params=dict(
            excess=dict(type="linear_quantile", slope=0.20, intercept=3.05),
            robust_sharpe=dict(type="linear_quantile", slope=78.95, intercept=-10.0),
            r_cubed=dict(type="linear_quantile", slope=42.21, intercept=-5.13),
        ),
    ),
    "01810.HK": dict(  # 小米
        weights=dict(excess=0.404,  robust_sharpe=0.411,  r_cubed=0.185),
        scoring_params=dict(
            excess=dict(type="linear_quantile", slope=0.60, intercept=56.41),
            robust_sharpe=dict(type="linear_quantile", slope=155.17, intercept=5.0),
            r_cubed=dict(type="linear_quantile", slope=225.0, intercept=5.0),
        ),
    ),
    "06181.HK": dict(  # 老铺
        weights=dict(excess=0.392,  robust_sharpe=0.323,  r_cubed=0.285),
        scoring_params=dict(
            excess=dict(type="linear_quantile", slope=1.46, intercept=450.43),
            robust_sharpe=dict(type="linear_quantile", slope=118.42, intercept=-31.71),
            r_cubed=dict(type="linear_quantile", slope=5.07, intercept=-11.78),
        ),
    ),
    "000625.SZ": dict(  # 长安
        weights=dict(excess=0.3764, robust_sharpe=0.3358, r_cubed=0.2878),
        scoring_params=dict(
            excess=dict(type="linear_quantile", slope=1.0949, intercept=23.8280),
            robust_sharpe=dict(type="linear_quantile", slope=169.8113, intercept=79.7170),
            r_cubed=dict(type="linear_quantile", slope=204.5455, intercept=74.5455),
        ),
    ),
}


def get_scorer(ts_code: str) -> AbsoluteScorer:
    """获取指定标的的绝对评分器。"""
    if ts_code not in SCORING_CONFIG:
        raise KeyError(f"未配置 {ts_code} 的评分配置，请在 SCORING_CONFIG 中添加")
    return AbsoluteScorer(SCORING_CONFIG[ts_code])

# 单参数扫描定义（围绕最优值展开）
SWEEP_DEFS = {
    "pm_grid_down": {
        "values": [0.04,0.045, 0.05,0.055, 0.06,0.065,0.07,0.075, 0.08,0.085,0.09,0.095,0.1,0.105,0.11,0.115,0.12,0.13,0.14,0.15],
        "label": PARAM_LABELS.get("pm_grid_down", "pm_grid_down"),
    },
    "pm_grid_up": {
        "values": [0.025,0.026,0.027,0.028,0.029,0.03,0.031,0.032,0.033,0.034, 0.035,0.036,0.037,0.038,0.039, 0.04,0.041,0.042,0.043,0.044,0.045,0.046,0.047,0.048,0.049, 0.05,0.055, 0.06,0.065,0.07,0.075, 0.08,0.085,0.09,0.095,0.10,0.105,0.11],
        "label": PARAM_LABELS.get("pm_grid_up", "pm_grid_up"),
    },
    "pm_reduce_pct": {
        "values": [0.31,0.32,0.33,0.34, 0.35,0.36,0.37,0.38,0.39, 0.4,0.45,0.46,0.47,0.48,0.49,0.50,0.51,0.52,0.53,0.54,0.55,0.56,0.57,0.58,0.59, 0.60,0.61,0.62,0.63,0.64,0.65,0.66,0.67,0.68,0.69,0.70,0.71,0.72,0.73,0.74,0.75,0.76,0.77,0.78,0.79,0.80],
        "label": PARAM_LABELS.get("pm_reduce_pct", "pm_reduce_pct"),
    },
    "pm_add_pct": {
        "values": [0.20,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29,0.30,0.31,0.32,0.33,0.34,0.35,0.36,0.37,0.38,0.39,0.40,0.41,0.42,0.43,0.44,0.45,0.46,0.47,0.48,0.49,0.50,0.51,0.52,0.53,0.54,0.55],
        "label": PARAM_LABELS.get("pm_add_pct", "pm_add_pct"),
    },
    "pm_adx_threshold": {
        "values": [15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45],
        "label": PARAM_LABELS.get("pm_adx_threshold", "pm_adx_threshold"),
    },
    "pm_agg_k":{
        "values": [0.60,0.61,0.62,0.63,0.64,0.65,0.66,0.67,0.68,0.69,0.70,0.71,0.72,0.73,0.74,0.75,0.76,0.77,0.78,0.79,0.80,0.81,0.82,0.83,0.84,0.85,0.86,0.87,0.88,0.89,0.90,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99,1.0],
        "label": PARAM_LABELS.get("pm_agg_k", "pm_agg_k"),
    }
}

# 滚动窗口参数
WINDOW_BARS = 252
STEP_BARS = 10
WARMUP = 20

# 并行工作进程数
WORKERS = max(1, multiprocessing.cpu_count() - 1)

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


def _split_batches(items: list, n_batches: int) -> list[list]:
    """将列表分割为 n_batches 个大致均匀的子列表。"""
    if n_batches <= 1 or len(items) <= n_batches:
        return [items]
    k, m = divmod(len(items), n_batches)
    batches = []
    start = 0
    for i in range(n_batches):
        end = start + k + (1 if i < m else 0)
        if end > start:
            batches.append(items[start:end])
        start = end
    return batches


# ═══════════════════════════════════════════════════════════════
# 多进程 Worker 函数（模块级别，确保可 pickle）
# ═══════════════════════════════════════════════════════════════

def _worker_stage_a(args):
    """Stage A worker: 全量历史回测一批参数。"""
    params_batch, base_kwargs, df, benchmark_return = args
    results = []
    for params in params_batch:
        stats = run_one_backtest(params, base_kwargs, df, benchmark_return)
        if stats:
            stats["_window"] = "full"
            stats["_params"] = params
            results.append(stats)
    return results


def _worker_stage_b(args):
    """Stage B worker: 滚动窗口回测一批参数。"""
    params_batch, base_kwargs, df, warmup, window_bars, step_bars = args
    results = []
    dates = pd.to_datetime(df["Date"])
    n = len(df)
    for params in params_batch:
        for start_idx in range(warmup, n - window_bars, step_bars):
            end_idx = start_idx + window_bars
            warmup_start = max(0, start_idx - warmup)
            df_win = df.iloc[warmup_start:end_idx]
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
                results.append(stats)
    return results


def _worker_stage_c(args):
    """Stage C worker: 8种场景模拟回测一批参数。"""
    params_batch, base_kwargs, scenario_dfs, scenario_benchmarks, sim_start = args
    results = []
    for params in params_batch:
        for sc_label, sc_df in scenario_dfs.items():
            sc_kwargs = {**base_kwargs, "pm_start_date": sim_start}
            stats = run_one_backtest(params, sc_kwargs, sc_df,
                                     benchmark_return=scenario_benchmarks[sc_label])
            if stats and stats.get("总交易", 0) > 0:
                stats["_window"] = f"sim_{sc_label}"
                stats["_params"] = params
                results.append(stats)
    return results


def _run_three_windows(param_combos: list[dict], base_kwargs: dict,
                       df_base: pd.DataFrame, scenario_dfs: dict,
                       scenario_benchmarks: dict, sim_start: str,
                       full_bm: float) -> list[dict]:
    """对一组参数组合运行三类窗口回测（多进程并行），返回原始结果列表。"""
    all_results = []
    n_workers = min(WORKERS, len(param_combos)) if param_combos else 1

    # Stage A: 全量历史
    batches = _split_batches(param_combos, n_workers)
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(_worker_stage_a,
                                   (batch, base_kwargs, df_base, full_bm))
                   for batch in batches]
        for future in as_completed(futures):
            all_results.extend(future.result())

    # Stage B: 滚动窗口
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(_worker_stage_b,
                                   (batch, base_kwargs, df_base,
                                    WARMUP, WINDOW_BARS, STEP_BARS))
                   for batch in batches]
        for future in as_completed(futures):
            all_results.extend(future.result())

    # Stage C: 8场景模拟
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(_worker_stage_c,
                                   (batch, base_kwargs, scenario_dfs,
                                    scenario_benchmarks, sim_start))
                   for batch in batches]
        for future in as_completed(futures):
            all_results.extend(future.result())

    return all_results


def _debug_cross_check(all_results: list[dict], sweep_param: str,
                       optimal_params: dict):
    """打印 OPTIMAL_PARAMS 组合的详细得分分解，用于跨参数扫描对比。"""
    # 找到与 OPTIMAL_PARAMS 完全匹配的组合
    target_key = tuple(optimal_params.get(p) for p in SWEEP_DEFS)
    for r in all_results:
        r_key = tuple(r["_params"].get(p) for p in SWEEP_DEFS)
        if r_key == target_key:
            break
    else:
        return  # 当前扫描中未测试此组合

    # 收集该组合的所有窗口结果
    from collections import defaultdict
    group = [r for r in all_results
             if tuple(r["_params"].get(p) for p in SWEEP_DEFS) == target_key]

    full_r = next((r for r in group if r["_window"] == "full"), None)
    rolling_rs = [r for r in group if r["_window"].startswith("rolling_")]
    sim_rs = [r for r in group if r["_window"].startswith("sim_")]

    full_score = full_r["score"] if full_r else 0
    rolling_median = float(np.median([r["score"] for r in rolling_rs])) if rolling_rs else 0
    sim_weighted = 0.0
    sim_probs_total = 0.0
    probs_cfg = get_scenario_probs(BASE_KWARGS["ts_code"])
    for r in sim_rs:
        sc_label = r["_window"].replace("sim_", "")
        prob = probs_cfg.get(sc_label, 0.0)
        if prob > 0:
            sim_weighted += r["score"] * prob
            sim_probs_total += prob
    sim_weighted = sim_weighted / sim_probs_total if sim_probs_total > 0 else 0
    combined = 0.5 * rolling_median + 0.3 * sim_weighted + 0.2 * full_score

    print(f"  🔍 [DEBUG cross-check] sweep={sweep_param} "
          f"params=({','.join(f'{PARAM_LABELS.get(p,p)}={optimal_params.get(p)}' for p in SWEEP_DEFS)})")
    print(f"     full_score={full_score:.2f}  "
          f"rolling_median={rolling_median:.2f} (n={len(rolling_rs)})  "
          f"sim_weighted={sim_weighted:.2f} (n={len(sim_rs)})")
    print(f"     combined = 0.5×{rolling_median:.2f} + 0.3×{sim_weighted:.2f} + 0.2×{full_score:.2f} = {combined:.2f}")
    if full_r:
        print(f"     raw: excess={full_r.get('excess',0):.2f}  "
              f"robust_sharpe={full_r.get('robust_sharpe',0):.2f}  "
              f"r_cubed={full_r.get('r_cubed',0):.2f}  "
              f"trades={full_r.get('总交易',0)}")


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
        probs_cfg = get_scenario_probs(BASE_KWARGS["ts_code"])
        for r in group:
            if r["_window"].startswith("sim_"):
                sc_label = r["_window"].replace("sim_", "")
                prob = probs_cfg.get(sc_label, 0.0)
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
        all_scored = {}

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

            # 绝对评分（固定尺度，不依赖同批次其他组合）
            t0 = time.time()
            raw_results = [r for r in raw_results if r.get("总交易", 0) > 0]
            for r in raw_results:
                r.setdefault("excess", r.get("超额收益 (%)", 0.0) or 0.0)
                r.setdefault("robust_sharpe", r.get("稳健夏普比率", 0.0) or 0.0)
                r.setdefault("r_cubed", r.get("R立方", 0.0) or 0.0)
            scorer = get_scorer(BASE_KWARGS["ts_code"])
            scorer.score_batch(raw_results)
            scored_results = raw_results
            weights = scorer.weights
            print(f"  绝对评分完成: { {k:round(v,3) for k,v in weights.items()} }")

            # 汇总
            summary = _aggregate_sweep(scored_results, pname)

            # ── debug: 打印 OPTIMAL_PARAMS 交叉组合的详细得分分解 ──
            # _debug_cross_check(scored_results, pname, OPTIMAL_PARAMS)

            for s in summary:
                print(f"    {label}={s['value']:.3f} → 综合={s['score']:.2f}  "
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
            old_entry = next(s for s in summary if s["value"] == old_val)
            old_score = old_entry["score"]
            print(f"  ✅ {pname} 最优值: {best_val:.3f} (得分{best_score:.2f}),上一轮最优值: {old_val:.3f}(得分{old_score:.2f}) ")
            if abs(best_val - old_val) > 0.001 and round(best_score - old_score, 2) > 0.02:
                print(f"  🔄 {pname} 最优值改变: {old_val:.3f}(得分{old_score:.2f}) → "
                      f"{best_val:.3f}(得分{best_score:.2f})")
                OPTIMAL_PARAMS[pname] = round(best_val, 3)
                param_changed = True
                break  # 停止本轮扫描，用新参数重新开始
            else:
                print(f"  ✅ {pname} 最优值未变: {old_val:.3f} (得分{old_score:.2f})")

        # ── 收敛判断 ──
        if not param_changed:
            # 完整扫描一轮且无任何参数改变 → 收敛
            all_sweeps["_base_params"] = OPTIMAL_PARAMS
            print(f"\n[{_now()}] ✅ 第 {round_num} 轮收敛！最优参数不再改变。")
            break
        else:
            print(f"\n[{_now()}] ⚠️ 最优参数已改变，重新扫描全部参数...")

    ts_code_clean = BASE_KWARGS["ts_code"].replace(".", "_")

    # ── 最终绘图：5张图合并为一个文件 ──
    _plot_sweep_curves(
        all_sweeps, out_dir,
        filename=f"turtle_sweep_all_{ts_code_clean}.png",
        title=f"单参数扫描 — 收敛后 (gd={OPTIMAL_PARAMS['pm_grid_down']:.3f} "
              f"gu={OPTIMAL_PARAMS['pm_grid_up']:.3f} "
              f"rp={OPTIMAL_PARAMS['pm_reduce_pct']:.3f} "
              f"ap={OPTIMAL_PARAMS['pm_add_pct']:.3f} "
              f"adx={int(OPTIMAL_PARAMS['pm_adx_threshold'])})",
        base_params=OPTIMAL_PARAMS,
    )

    # ── 保存最优参数三窗口明细 Excel ──
    target_params = {p: OPTIMAL_PARAMS[p] for p in SWEEP_DEFS}
    optimal_results = [r for r in scored_results
                       if all(r["_params"].get(p, 0) == target_params[p]
                              for p in SWEEP_DEFS)]

    _DETAIL_COLS = [
        ("窗口", "_window"),
        ("综合得分", "score"),
        ("超额收益 (%)", "excess"),
        ("稳健夏普比率", "robust_sharpe"),
        ("R立方", "r_cubed"),
        ("CAGR (%)", "CAGR (%)"),
        ("最大回撤 (%)", "最大回撤 (%)"),
        ("总收益率 (%)", "总收益率 (%)"),
        ("夏普比率", "夏普比率"),
        ("胜率 (%)", "胜率 (%)"),
        ("总交易", "总交易"),
        ("donchian交易", "donchian交易"),
        ("donchian盈利次数", "donchian盈利次数"),
        ("donchian强制平仓", "donchian强制平仓"),
        ("R平方", "R平方"),
        ("MAR比率", "MAR比率"),
    ]

    def _build_detail_sheet(rows):
        data = []
        for r in rows:
            row_data = {}
            # 最优参数组合
            for pname in SWEEP_DEFS:
                row_data[SWEEP_DEFS[pname]["label"]] = OPTIMAL_PARAMS[pname]
            for label, key in _DETAIL_COLS:
                val = r.get(key)
                row_data[label] = round(val, 4) if isinstance(val, float) else val
            data.append(row_data)
        return pd.DataFrame(data)

    detail_file = f"turtle_optimal_details_{ts_code_clean}.xlsx"
    with pd.ExcelWriter(os.path.join(out_dir, detail_file),
                        engine="openpyxl") as detail_writer:
        full_rows = [r for r in optimal_results if r["_window"] == "full"]
        rolling_rows = sorted(
            [r for r in optimal_results if r["_window"].startswith("rolling_")],
            key=lambda r: r["_window"])
        sim_rows = [r for r in optimal_results if r["_window"].startswith("sim_")]

        if full_rows:
            _build_detail_sheet(full_rows).to_excel(
                detail_writer, sheet_name="全量历史", index=False)
        if rolling_rows:
            _build_detail_sheet(rolling_rows).to_excel(
                detail_writer, sheet_name="滚动窗口", index=False)
        if sim_rows:
            _build_detail_sheet(sim_rows).to_excel(
                detail_writer, sheet_name="场景模拟", index=False)

    print(f"  最优参数明细: result/{detail_file} "
          f"(全量{len(full_rows)} 滚动{len(rolling_rows)} 场景{len(sim_rows)})")

    # ── 保存 Excel（所有参数汇总到一个文件）──
    with pd.ExcelWriter(os.path.join(out_dir, f"turtle_sweep_all_{ts_code_clean}.xlsx"),
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
    print(f"  图表: result/turtle_sweep_all_{ts_code_clean}.png")
    print(f"  Excel: result/turtle_sweep_all_{ts_code_clean}.xlsx")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="海龟策略单参数扫描")
    parser.add_argument("--ts-code", type=str, default="09992.HK",
                        help=f"股票代码 (默认: 09992.HK)")
    parser.add_argument("--workers", type=int, default=WORKERS,
                        help=f"并行工作进程数 (默认: {WORKERS})")
    parser.add_argument("--serial", action="store_true",
                        help="强制串行执行（不使用多进程）")
    args = parser.parse_args()
    if args.serial:
        WORKERS = 1
    else:
        WORKERS = max(1, args.workers)
    init_config(args.ts_code)
    main()
