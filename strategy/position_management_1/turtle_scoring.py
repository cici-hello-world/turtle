#!/usr/bin/env python3
"""海龟策略公平评分 — 熵权法 + 三窗口统一评分

基于对 09992.HK 回测数据的全方位分析，用数据驱动方式评选最优参数组合。

评分流程：
  1. 所有参数组合 × 三类窗口回测（全量历史 / 滚动252天 / 8场景模拟）
  2. 合并结果 → 剔除0交易 → 熵权法计算指标权重
  3. 统一百分位归一化 → "超出第二名比率"奖惩
  4. 按参数汇总 → 综合 = 0.5×滚动中位数 + 0.3×模拟加权 + 0.2×全量得分

用法：
  python strategy/position_management_1/turtle_scoring.py
"""

import copy
import itertools
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import timedelta
import multiprocessing

import numpy as np
import pandas as pd

# 项目路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from core.config import BacktestConfig
from core.data import fetch_data
from core.indicators import compute_indicators
from core.engine import run_backtest
from core.statistics import calc_stats
from core.scoring import (compute_percentile_scores,
                          apply_reward_penalty)

# 复用已有函数
from strategy.position_management_1.turtle_optimize import (
    run_one_backtest, _map_stats_keys,
)
from strategy.position_management_1.grid_search_pm import (
    prepare_scenario_dfs, SCENARIO_LABEL_MAP, get_scenario_probs,
)
from strategy.position_management_1.simulate_market_scenarios import (
    get_last_price as sim_get_last_price,
)

# ═══════════════════════════════════════════════════════════════
# 配置 — 全部从共享配置加载
# ═══════════════════════════════════════════════════════════════

from strategy.position_management_1 import common_config

# 本地激活的配置（由 init_config() 设置）
BASE_KWARGS = None
SWEEP_RANGES = None

# 从共享配置导入常量
WINDOW_BARS = common_config.WINDOW_BARS
STEP_BARS = common_config.STEP_BARS
WARMUP = common_config.WARMUP
COMBINED_SCORE_WEIGHTS = common_config.COMBINED_SCORE_WEIGHTS
_constraint_ok = common_config._constraint_ok
get_scoring_weights = common_config.get_scoring_weights


def init_config(ts_code: str):
    """根据 ts_code 激活对应的配置。"""
    global BASE_KWARGS, SWEEP_RANGES
    common_config.init_config(ts_code)
    BASE_KWARGS = common_config.BASE_KWARGS
    SWEEP_RANGES = common_config.SWEEP_RANGES_ACTIVE

# 并行工作进程数（可通过命令行 --workers N 覆盖）
WORKERS = max(1, multiprocessing.cpu_count() - 1)

# ═══════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════

def _params_to_key(params: dict) -> tuple:
    """参数 → 去重 key"""
    return tuple(params[p] for p in SWEEP_RANGES)


def _compute_buy_hold(df: pd.DataFrame, start_date: str) -> float:
    """计算同期买入持有收益率 (%)"""
    mask = df["Date"] >= start_date
    close = df.loc[mask, "Close"]
    if len(close) < 2:
        return 0.0
    return round((close.iloc[-1] / close.iloc[0] - 1) * 100, 2)


def _now():
    return time.strftime("%H:%M:%S")


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
    """Stage A worker: 对一批参数组合在全量历史上回测。"""
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
    """Stage B worker: 对一批参数组合在所有滚动窗口上回测。"""
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
    """Stage C worker: 对一批参数组合在所有模拟场景上回测。"""
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


# ═══════════════════════════════════════════════════════════════
# Bootstrap 敏感性分析
# ═══════════════════════════════════════════════════════════════

def _perturb_sweep_ranges(sweep_ranges: dict, rng: np.random.RandomState) -> dict:
    """随机扰动 SWEEP_RANGES：每个参数随机保留 75-90% 的候选值。"""
    perturbed = {}
    for pname, values in sweep_ranges.items():
        if len(values) <= 2:
            perturbed[pname] = list(values)
        else:
            keep_frac = rng.uniform(0.75, 0.90)
            n_keep = max(2, int(len(values) * keep_frac))
            kept = sorted(rng.choice(values, size=n_keep, replace=False).tolist())
            perturbed[pname] = kept
    return perturbed


def _generate_param_combos(sweep_ranges: dict) -> list[dict]:
    """从 SWEEP_RANGES 生成参数组合列表（笛卡尔积 + 约束过滤）。"""
    import itertools
    param_names = list(sweep_ranges.keys())
    combos = []
    for values in itertools.product(*sweep_ranges.values()):
        params = dict(zip(param_names, values))
        if _constraint_ok(params):
            combos.append(params)
    return combos


def bootstrap_sensitivity(base_kwargs: dict, df_base: pd.DataFrame,
                           full_bm: float, n_iter: int = 20,
                           n_workers: int = 1, seed: int = 42) -> dict:
    """Bootstrap 敏感性分析：多次扰动 SWEEP_RANGES，评估最优参数和熵权的稳定性。

    对每次扰动：
      1. 随机丢弃 SWEEP_RANGES 中每个参数 10-25% 的候选值
      2. 生成新的参数组合集合
      3. 仅跑 Stage A（全量历史回测），计算熵权和百分位排名
      4. 记录最优参数和熵权

    最后汇总所有迭代的结果，报告各参数最优值的中位数、标准差、变异系数。

    Args:
        base_kwargs: 基础回测参数
        df_base: 行情数据
        full_bm: 全量买入持有基准收益
        n_iter: 扰动次数（建议 20-50）
        n_workers: 并行进程数
        seed: 基础随机种子

    Returns:
        {
            "param_stability": {pname: {median, std, cv, p25, p75, all_values}},
            "weight_stability": {metric: {mean, std, all_values}},
            "rank_stability": {pname: {top1_freq: {val: count}}},
            "all_best_params": [...],
            "all_weights": [...],
        }
    """
    print(f"\n{'='*60}")
    print(f"🔬 Bootstrap 敏感性分析 ({n_iter} 次扰动)")
    print(f"{'='*60}")
    print(f"  方法: 每次随机丢弃 SWEEP_RANGES 中 10-25% 候选值")
    print(f"  范围: Stage A 全量历史 + 熵权 + 百分位排名")
    print(f"  目标: 评估最优参数和权重的稳定性\n")

    all_best_params = []  # [{pname: val, "score": ...}, ...]
    all_weights = []      # [{excess: w, ...}, ...]
    rng = np.random.RandomState(seed)

    for iteration in range(n_iter):
        iter_seed = rng.randint(0, 2**31 - 1)
        iter_rng = np.random.RandomState(iter_seed)

        # 扰动 SWEEP_RANGES
        perturbed_ranges = _perturb_sweep_ranges(SWEEP_RANGES, iter_rng)
        combos = _generate_param_combos(perturbed_ranges)

        # 仅 Stage A：全量历史回测
        from strategy.position_management_1.turtle_optimize import (
            run_one_backtest, _map_stats_keys,
        )
        results = []
        for params in combos:
            stats = run_one_backtest(params, base_kwargs, df_base,
                                     benchmark_return=full_bm)
            if stats and stats.get("总交易", 0) > 0:
                stats = _map_stats_keys(stats)
                stats["_params"] = params
                results.append(stats)

        if len(results) < 3:
            continue

        # 使用固定权重
        weights = get_scoring_weights(base_kwargs["ts_code"])
        all_weights.append(weights)

        # 百分位排名
        compute_percentile_scores(results, weights)

        # 最优参数
        best = max(results, key=lambda r: r["score"])
        best_params = {k: best["_params"].get(k, None) for k in SWEEP_RANGES}
        best_params["score"] = best["score"]
        best_params["_iter"] = iteration
        all_best_params.append(best_params)

        # 进度
        if (iteration + 1) % 5 == 0 or iteration == 0:
            print(f"  [{iteration + 1}/{n_iter}] "
                  f"combos={len(combos)}→有效{len(results)}  "
                  f"权重: excess={weights['excess']:.3f} "
                  f"robust_sharpe={weights['robust_sharpe']:.3f} "
                  f"r_cubed={weights['r_cubed']:.3f}  "
                  f"最优综合分={best['score']:.1f}")

    if len(all_best_params) < 3:
        print("  ⚠️ 有效迭代不足，无法输出稳定性报告")
        return {}

    # ── 汇总统计 ──
    df_best = pd.DataFrame(all_best_params)
    df_w = pd.DataFrame(all_weights)

    print(f"\n  📊 参数稳定性报告 ({len(all_best_params)} 次有效扰动)")
    print(f"  {'='*80}")
    print(f"  {'参数':<22} {'中位数':>8} {'标准差':>8} {'CV':>7} {'P25':>8} {'P75':>8} {'范围':>12}")
    print(f"  {'─'*80}")

    param_stability = {}
    param_labels = {
        "pm_grid_down": "gd", "pm_grid_up": "gu",
        "pm_reduce_pct": "rp", "pm_add_pct": "ap",
        "pm_adx_threshold": "adx", "pm_agg_k": "k",
    }
    for pname in SWEEP_RANGES:
        vals = df_best[pname].dropna()
        if len(vals) < 3:
            continue
        median = float(vals.median())
        std = float(vals.std())
        cv = std / abs(median) if abs(median) > 0.001 else float('inf')
        p25 = float(vals.quantile(0.25))
        p75 = float(vals.quantile(0.75))

        param_stability[pname] = {
            "median": round(median, 4),
            "std": round(std, 4),
            "cv": round(cv, 4),
            "p25": round(p25, 4),
            "p75": round(p75, 4),
            "all_values": [round(v, 4) for v in vals.tolist()],
        }

        label = param_labels.get(pname, pname)
        cv_flag = ""
        if cv > 0.15:
            cv_flag = " ⚠️高变异"
        elif cv > 0.05:
            cv_flag = " ⚡中等"
        else:
            cv_flag = " ✅稳定"
        print(f"  {label:<22} {median:>8.3f} {std:>8.3f} {cv:>7.3f}"
              f" {p25:>8.3f} {p75:>8.3f}"
              f" [{vals.min():.3f}, {vals.max():.3f}]{cv_flag}")

    print(f"\n  📊 熵权稳定性报告")
    print(f"  {'─'*50}")
    print(f"  {'指标':<20} {'均值':>8} {'标准差':>8} {'CV':>7}")
    print(f"  {'─'*50}")

    weight_stability = {}
    for metric in ["excess", "robust_sharpe", "r_cubed"]:
        w_vals = df_w[metric].dropna()
        mean_w = float(w_vals.mean())
        std_w = float(w_vals.std())
        cv_w = std_w / mean_w if mean_w > 0.001 else float('inf')
        weight_stability[metric] = {
            "mean": round(mean_w, 4),
            "std": round(std_w, 4),
            "cv": round(cv_w, 4),
            "all_values": [round(v, 4) for v in w_vals.tolist()],
        }
        cv_flag = " ⚠️" if cv_w > 0.15 else (" ⚡" if cv_w > 0.05 else " ✅")
        print(f"  {metric:<20} {mean_w:>8.3f} {std_w:>8.3f} {cv_w:>7.3f}{cv_flag}")

    # ── 排名稳定性：最优参数各值出现频率 ──
    print(f"\n  📊 排名稳定性 — 各参数值成为最优的频率 (Top 3)")
    rank_stability = {}
    for pname in SWEEP_RANGES:
        label = param_labels.get(pname, pname)
        value_counts = df_best[pname].value_counts().head(3)
        rank_stability[pname] = {
            "top1_freq": {str(k): int(v) for k, v in value_counts.items()}
        }
        items = []
        for val, count in value_counts.items():
            pct = count / len(df_best) * 100
            items.append(f"{val:.3f}({pct:.0f}%)")
        print(f"  {label:<22} {', '.join(items)}")

    # ── 建议 ──
    print(f"\n  📋 诊断与建议:")
    high_cv_params = [p for p, s in param_stability.items()
                      if s["cv"] > 0.10]
    high_cv_weights = [m for m, s in weight_stability.items()
                       if s["cv"] > 0.10]

    if high_cv_params:
        labels = [param_labels.get(p, p) for p in high_cv_params]
        print(f"  ⚠️ 高变异参数 (CV>0.10): {', '.join(labels)}")
        print(f"     → 建议: 这些参数的最优值对 sweep 网格敏感，")
        print(f"            改用连续优化 (turtle_optuna.py) 确认")

    if high_cv_weights:
        print(f"  ⚠️ 高变异权重 (CV>0.10): {', '.join(high_cv_weights)}")
        print(f"     → 建议: 运行 core/calibrate.py 在全域采样上固定熵权")

    if not high_cv_params and not high_cv_weights:
        print(f"  ✅ 所有参数和权重 CV < 0.10，评分体系对 SWEEP_RANGES 稳健")

    return {
        "param_stability": param_stability,
        "weight_stability": weight_stability,
        "rank_stability": rank_stability,
        "all_best_params": all_best_params,
        "all_weights": all_weights,
        "n_iterations": len(all_best_params),
    }


# ═══════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  海龟策略公平评分 — 熵权法 + 三窗口统一评分             ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()
    print(f"  股票: {BASE_KWARGS['stock_name']} ({BASE_KWARGS['ts_code']})")
    print(f"  参数: {list(SWEEP_RANGES.keys())}")
    total_combos = 1
    for v in SWEEP_RANGES.values():
        total_combos *= len(v)
    print(f"  原始组合数: {total_combos}")

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
        pm_bb_period=config0.pm_bb_period, pm_bb_k=config0.pm_bb_k,
        pm_bb_width_lookback=config0.pm_bb_width_lookback,
    )
    print(f"   数据: {len(df_base)} 条  "
          f"({df_base['Date'].iloc[0]} ~ {df_base['Date'].iloc[-1]})")

    # ── 生成参数组合 ──
    param_names = list(SWEEP_RANGES.keys())
    param_combos = []
    for values in itertools.product(*SWEEP_RANGES.values()):
        params = dict(zip(param_names, values))
        if _constraint_ok(params):
            param_combos.append(params)
    print(f"   约束后有效: {len(param_combos)} 组\n")

    # ── 全量历史基准收益 ──
    full_bm = _compute_buy_hold(df_base, BASE_KWARGS["pm_start_date"])
    print(f"  买入持有基准 (自{BASE_KWARGS['pm_start_date']}): {full_bm:+.2f}%")

    # ═══════════════════════════════════════════════════════
    # Stage A: 全量历史回测
    # ═══════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"[{_now()}] 🅰 Stage A: 全量历史回测 ({len(param_combos)} 组, "
          f"{WORKERS} workers)")
    print(f"{'='*60}")

    all_results = []  # 所有窗口×所有参数的结果
    t0 = time.time()

    batches = _split_batches(param_combos, WORKERS)
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        futures = {
            executor.submit(_worker_stage_a,
                            (batch, BASE_KWARGS, df_base, full_bm)): idx
            for idx, batch in enumerate(batches)
        }
        for future in as_completed(futures):
            batch_results = future.result()
            all_results.extend(batch_results)
            done = sum(1 for f in futures if f.done())
            elapsed = time.time() - t0
            print(f"    [{done}/{len(batches)} batches] {elapsed:.0f}s  "
                  f"累计有效: {len(all_results)}")

    full_count = len(all_results)
    print(f"  全量完成: {full_count} 组有效  "
          f"({len(param_combos)-full_count} 组无交易)  "
          f"耗时 {(time.time()-t0)/60:.1f}min")

    # ═══════════════════════════════════════════════════════
    # Stage B: 真实滚动窗口回测
    # ═══════════════════════════════════════════════════════

    dates = pd.to_datetime(df_base["Date"])
    n = len(df_base)
    n_windows = max(0, (n - WARMUP - WINDOW_BARS) // STEP_BARS + 1)

    print(f"\n{'='*60}")
    print(f"[{_now()}] 🅱 Stage B: 滚动窗口回测 "
          f"({len(param_combos)}组 × ~{n_windows}窗口, {WORKERS} workers)")
    print(f"{'='*60}")
    print(f"  窗口: {WINDOW_BARS}天 × 步长{STEP_BARS}天 × 预热{WARMUP}天")

    t0 = time.time()
    rolling_count = 0

    batches = _split_batches(param_combos, WORKERS)
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        futures = {
            executor.submit(_worker_stage_b,
                            (batch, BASE_KWARGS, df_base,
                             WARMUP, WINDOW_BARS, STEP_BARS)): idx
            for idx, batch in enumerate(batches)
        }
        for future in as_completed(futures):
            batch_results = future.result()
            all_results.extend(batch_results)
            rolling_count += len(batch_results)
            done = sum(1 for f in futures if f.done())
            elapsed = time.time() - t0
            print(f"    [{done}/{len(batches)} batches] {elapsed:.0f}s  "
                  f"累计: {len(all_results)}")

    print(f"  滚动完成: +{rolling_count} 条  (总累计 {len(all_results)})  "
          f"耗时 {(time.time()-t0)/60:.1f}min")

    # ═══════════════════════════════════════════════════════
    # Stage C: 8种场景模拟回测
    # ═══════════════════════════════════════════════════════
    if COMBINED_SCORE_WEIGHTS.get("sim_weighted", 0) == 0:
        print(f"\n⏭️  跳过 Stage C: 模拟场景回测 "
              f"(sim_weighted=0，模拟数据不参与综合评分)")
    else:
        print(f"\n{'='*60}")
        print(f"[{_now()}] 🅲 Stage C: 8种场景模拟回测 "
              f"({len(param_combos)}组 × 8场景, {WORKERS} workers)")
        print(f"{'='*60}")

        t0 = time.time()

        # 获取最后价格和日期
        try:
            last_price, last_date = sim_get_last_price()
        except Exception:
            from core.cache import load as cache_load
            cached = cache_load(config0.ts_code, config0.market)
            last_price = float(cached["Close"].iloc[-1])
            last_date = cached["Date"].iloc[-1]
        sim_start = (last_date + timedelta(days=1)).strftime("%Y%m%d")
        print(f"  模拟起始: {sim_start}, 最新价: {last_price:.2f}")

        # 生成场景数据
        scenario_dfs, scenario_benchmarks = prepare_scenario_dfs(
            last_price, last_date, config0, df_base)

        sim_count = 0
        batches = _split_batches(param_combos, WORKERS)
        with ProcessPoolExecutor(max_workers=WORKERS) as executor:
            futures = {
                executor.submit(_worker_stage_c,
                                (batch, BASE_KWARGS, scenario_dfs,
                                 scenario_benchmarks, sim_start)): idx
                for idx, batch in enumerate(batches)
            }
            for future in as_completed(futures):
                batch_results = future.result()
                all_results.extend(batch_results)
                sim_count += len(batch_results)
                done = sum(1 for f in futures if f.done())
                elapsed = time.time() - t0
                print(f"    [{done}/{len(batches)} batches] {elapsed:.0f}s  "
                      f"累计: {len(all_results)}")

        print(f"  模拟完成: +{sim_count} 条  (总累计 {len(all_results)})  "
              f"耗时 {(time.time()-t0)/60:.1f}min")

    # ═══════════════════════════════════════════════════════
    # 评分
    # ═══════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"[{_now()}] 📊 统一评分")
    print(f"{'='*60}")
    print(f"  总结果: {len(all_results)} 条")

    # Step 1: 剔除总交易=0
    before = len(all_results)
    all_results = [r for r in all_results if r.get("总交易", 0) > 0]
    print(f"  剔除0交易: {before} → {len(all_results)}")

    # Step 2: 使用固定权重（来自校准配置，与 turtle_sweep.py 一致）
    weights = get_scoring_weights(BASE_KWARGS["ts_code"])
    print(f"  固定权重: excess={weights['excess']:.3f}  "
          f"robust_sharpe={weights['robust_sharpe']:.3f}  "
          f"r_cubed={weights['r_cubed']:.3f}")

    # Step 3: 映射指标 key（calc_stats 返回中文 key → scoring 需要英文 key）
    for r in all_results:
        r.setdefault("excess", r.get("超额收益 (%)", 0.0) or 0.0)
        r.setdefault("robust_sharpe", r.get("稳健夏普比率", 0.0) or 0.0)
        r.setdefault("r_cubed", r.get("R立方", 0.0) or 0.0)

    # Step 4: 统一百分位归一化（所有窗口拉通）
    compute_percentile_scores(all_results, weights)

    # Step 5: 奖惩机制
    apply_reward_penalty(all_results)

    # ═══════════════════════════════════════════════════════
    # 按参数组合汇总
    # ═══════════════════════════════════════════════════════
    print(f"\n[{_now()}] 📊 按参数组合汇总 ...")

    # 按参数分组
    from collections import defaultdict
    param_groups = defaultdict(list)
    for r in all_results:
        key = _params_to_key(r["_params"])
        param_groups[key].append(r)

    summary = []
    for key, group in param_groups.items():
        params = group[0]["_params"]

        # 全量窗口得分
        full_scores = [r["score"] for r in group if r["_window"] == "full"]
        full_score = full_scores[0] if full_scores else 50.0

        # 滚动窗口得分
        rolling_scores = [r["score"] for r in group
                          if r["_window"].startswith("rolling_")]
        if len(rolling_scores) >= 5:
            rolling_median = float(np.median(rolling_scores))
            rolling_p10 = float(np.percentile(rolling_scores, 10))
        else:
            rolling_median = float(np.mean(rolling_scores)) if rolling_scores else 50.0
            rolling_p10 = rolling_median

        # 场景模拟得分（概率加权）
        sim_scores = []
        sim_probs = []
        for r in group:
            if r["_window"].startswith("sim_"):
                sc_label = r["_window"].replace("sim_", "")
                prob = get_scenario_probs(BASE_KWARGS["ts_code"]).get(sc_label, 0.0)
                if prob > 0:
                    sim_scores.append(r["score"] * prob)
                    sim_probs.append(prob)

        if sim_scores and sum(sim_probs) > 0:
            sim_weighted = sum(sim_scores) / sum(sim_probs)
        else:
            sim_weighted = 50.0

        # 综合 = W_rolling × 滚动中位数 + W_sim × 模拟加权 + W_full × 全量
        combined = (COMBINED_SCORE_WEIGHTS["rolling_median"] * rolling_median +
                    COMBINED_SCORE_WEIGHTS["sim_weighted"] * sim_weighted +
                    COMBINED_SCORE_WEIGHTS["full_score"] * full_score)

        # 全量窗口原始指标
        full_orig = next((r for r in group if r["_window"] == "full"), group[0])

        summary.append({
            **params,
            "full_score": round(full_score, 2),
            "rolling_median": round(rolling_median, 2),
            "rolling_p10": round(rolling_p10, 2),
            "rolling_n": len(rolling_scores),
            "sim_weighted": round(sim_weighted, 2),
            "combined": round(combined, 2),
            "excess": round(full_orig.get("excess", 0), 2),
            "robust_sharpe": round(full_orig.get("robust_sharpe", 0), 2),
            "r_cubed": round(full_orig.get("r_cubed", 0), 2),
            "trades": full_orig.get("总交易", 0),
            "cagr": round(full_orig.get("CAGR (%)", 0), 2),
            "max_dd": round(full_orig.get("最大回撤 (%)", 0), 2),
        })

    summary.sort(key=lambda x: x["combined"], reverse=True)

    # ═══════════════════════════════════════════════════════
    # 输出
    # ═══════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"🏆 综合排名 TOP 20")
    print(f"{'='*70}")
    header = (f"{'#':<3} {'综合':>7} {'滚动中位':>8} {'模拟加权':>8} {'全量':>7} "
              f"{'P10':>7} {'超额%':>8} {'稳夏':>6} {'R³':>6}  gd   gu   rp   ap  adx")
    print(header)
    print(f"  {'─'*100}")
    for i, r in enumerate(summary[:20], 1):
        stable_flag = " ⚠️" if r["rolling_p10"] < 30 else ""
        print(f"  {i:<3} {r['combined']:>7.2f} {r['rolling_median']:>8.2f} "
              f"{r['sim_weighted']:>8.2f} {r['full_score']:>7.2f} "
              f"{r['rolling_p10']:>7.2f}{stable_flag} "
              f"{r['excess']:>8.2f} {r['robust_sharpe']:>6.2f} {r['r_cubed']:>6.2f}  "
              f"{r['pm_grid_down']:.2f} {r['pm_grid_up']:.2f} "
              f"{r['pm_reduce_pct']:.2f} {r['pm_add_pct']:.2f} "
              f"{r['pm_adx_threshold']:.0f}")

    # ── 最优参数 ──
    best = summary[0]
    common_config.update_optimal_params(BASE_KWARGS["ts_code"], best)

    print(f"\n  🥇 最优参数组合:")
    print(f"     pm_grid_down       = {best['pm_grid_down']:.2f}")
    print(f"     pm_grid_up         = {best['pm_grid_up']:.2f}")
    print(f"     pm_reduce_pct      = {best['pm_reduce_pct']:.2f}")
    print(f"     pm_add_pct         = {best['pm_add_pct']:.2f}")
    print(f"     pm_adx_threshold   = {int(best['pm_adx_threshold'])}")
    print(f"     pm_agg_k           = {best['pm_agg_k']:.1f}")
    print(f"     ──────────────────")
    print(f"     综合得分:  {best['combined']:.2f}")
    print(f"     滚动中位数: {best['rolling_median']:.2f}  P10={best['rolling_p10']:.2f}  "
          f"({'⚠️低稳定性' if best['rolling_p10']<30 else '✅稳定'})")
    print(f"     模拟加权:  {best['sim_weighted']:.2f}")
    print(f"     全量得分:  {best['full_score']:.2f}")
    print(f"     超额收益:  {best['excess']:+.2f}%")
    print(f"     稳健夏普:  {best['robust_sharpe']:.2f}")
    print(f"     R立方:     {best['r_cubed']:.2f}")
    print(f"     交易次数:  {best['trades']}")
    print(f"     CAGR:      {best['cagr']:.2f}%")
    print(f"     最大回撤:  {best['max_dd']:.2f}%")

    # ── 保存 Excel ──
    out_dir = os.path.join(PROJECT_ROOT, "result")
    os.makedirs(out_dir, exist_ok=True)

    ts_code_clean = BASE_KWARGS["ts_code"].replace(".", "_")

    xlsx_path = os.path.join(out_dir, f"turtle_scoring_result_{ts_code_clean}.xlsx")
    df_summary = pd.DataFrame(summary)
    df_summary.to_excel(xlsx_path, sheet_name="综合排名", index=False)
    print(f"\n📁 综合排名: {xlsx_path} ({len(summary)} 组)")

    # 每窗口明细
    detail_path = os.path.join(out_dir, f"turtle_scoring_detail_{ts_code_clean}.xlsx")
    detail_rows = []
    for r in all_results:
        p = r["_params"]
        detail_rows.append({
            "window": r["_window"],
            "gd": p["pm_grid_down"], "gu": p["pm_grid_up"],
            "rp": p["pm_reduce_pct"], "ap": p["pm_add_pct"],
            "adx": p["pm_adx_threshold"],
            "score": r["score"],
            "pct_excess": r.get("pct_excess", float("nan")),
            "pct_robust_sharpe": r.get("pct_robust_sharpe", float("nan")),
            "pct_r_cubed": r.get("pct_r_cubed", float("nan")),
            "excess": r.get("excess", 0),
            "robust_sharpe": r.get("robust_sharpe", 0),
            "r_cubed": r.get("r_cubed", 0),
            "trades": r.get("总交易", 0),
            "cagr": round(r.get("CAGR (%)", 0), 2),
            "max_dd": round(r.get("最大回撤 (%)", 0), 2),
        })
    pd.DataFrame(detail_rows).to_excel(detail_path, sheet_name="明细", index=False)
    print(f"📁 窗口明细: {detail_path} ({len(detail_rows)} 条)")

    print(f"\n[{_now()}] ✅ 评分完成！")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="海龟策略公平评分")
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
