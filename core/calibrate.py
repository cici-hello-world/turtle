#!/usr/bin/env python3
"""参数评分体系校准模块

通过在完整可行参数空间内随机采样，预先计算稳定的熵权和绝对评分阈值，
使后续任何 sweep/optimize 流程都使用统一的评分尺度，消除 SWEEP_RANGES
配置对最优参数筛选的影响。

核心思路：
  1. Latin Hypercube Sampling 在可行域均匀采样 N 个参数组合
  2. 跑全量历史回测，收集 excess / robust_sharpe / r_cubed
  3. 在全域样本上计算熵权（一次校准，长期固定）
  4. 计算各指标的分位数阈值（用于绝对评分）
  5. 持久化到 JSON，供 turtle_scoring / turtle_sweep / turtle_optuna 复用

用法：
  # 对 09992.HK 做校准
  python core/calibrate.py --ts-code 09992.HK --stock-name 泡泡玛特 \\
      --start-date 20201211 --pm-start-date 20210112

  # 对 03690.HK 做校准
  python core/calibrate.py --ts-code 03690.HK --stock-name 美团 \\
      --start-date 20181023 --pm-start-date 20180920
"""

import copy
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.config import BacktestConfig
from core.data import fetch_data
from core.indicators import compute_indicators
from core.scoring import entropy_weights, compute_weights

# 共享配置
from strategy.position_management_1 import common_config
from strategy.position_management_1.common_config import (
    DEFAULT_PARAM_SPACE, _constraint_ok, STOCKS,
)


def latin_hypercube_sample(param_space: dict, n_samples: int,
                            seed: int = 42) -> list[dict]:
    """用 Latin Hypercube Sampling 在参数空间均匀采样。

    相比纯随机采样，LHS 保证每个参数维度都均匀覆盖，
    用更少的样本获得更稳定的统计估计。

    如果 scipy 不可用，退化为均匀随机采样 + 拒绝采样。
    """
    rng = np.random.RandomState(seed)
    param_names = list(param_space.keys())
    n_dims = len(param_names)

    try:
        from scipy.stats.qmc import LatinHypercube
        sampler = LatinHypercube(d=n_dims, seed=seed)
        unit_samples = sampler.random(n=n_samples)
    except ImportError:
        unit_samples = rng.uniform(0, 1, size=(n_samples, n_dims))

    valid_samples = []
    max_extra = n_samples * 5  # 拒绝采样的最大尝试次数
    extra_count = 0

    for i in range(n_samples):
        params = {}
        for j, pname in enumerate(param_names):
            lo, hi = param_space[pname]
            val = lo + unit_samples[i, j] * (hi - lo)
            if pname == "pm_adx_threshold":
                val = int(round(val))
            else:
                val = round(val, 3)
            params[pname] = val

        if _constraint_ok(params):
            valid_samples.append(params)
        else:
            # 拒绝采样：生成额外样本补足
            while extra_count < max_extra:
                extra = rng.uniform(0, 1, size=n_dims)
                params2 = {}
                for j, pname in enumerate(param_names):
                    lo, hi = param_space[pname]
                    v = lo + extra[j] * (hi - lo)
                    if pname == "pm_adx_threshold":
                        v = int(round(v))
                    else:
                        v = round(v, 3)
                    params2[pname] = v
                extra_count += 1
                if _constraint_ok(params2):
                    valid_samples.append(params2)
                    break

    return valid_samples


def _worker_calibrate(args):
    """Worker 函数（模块级别，确保可 pickle）。

    对一批参数跑全量回测，返回原始指标值。
    """
    params_batch, base_kwargs, df, benchmark_return = args
    results = []
    for params in params_batch:
        try:
            from strategy.position_management_1.turtle_optimize import (
                run_one_backtest, _map_stats_keys,
            )
            stats = run_one_backtest(params, base_kwargs, df,
                                     benchmark_return=benchmark_return)
            if stats and stats.get("总交易", 0) > 0:
                stats = _map_stats_keys(stats)
                results.append({
                    "excess": float(stats.get("excess", 0) or 0),
                    "robust_sharpe": float(stats.get("robust_sharpe", 0) or 0),
                    "r_cubed": float(stats.get("r_cubed", 0) or 0),
                    "params": {k: v for k, v in params.items()},
                })
        except Exception:
            pass
    return results


def calibrate(base_kwargs: dict, df_base: pd.DataFrame,
              benchmark_return: float,
              param_space: dict | None = None,
              n_samples: int = 500,
              n_workers: int | None = None,
              seed: int = 42) -> dict:
    """主校准函数。

    在全域随机采样 → 全量回测 → 计算熵权 + 分位数阈值。

    Args:
        base_kwargs: 基础回测参数 (ts_code, start_date, pm_start_date, ...)
        df_base: 已计算技术指标的行情 DataFrame
        benchmark_return: 买入持有基准收益率 (%)
        param_space: 参数边界 dict，默认使用 DEFAULT_PARAM_SPACE
        n_samples: LHS 采样数量（建议 >= 300）
        n_workers: 并行进程数，默认 CPU-1
        seed: 随机种子

    Returns:
        calibration dict:
          - weights: {"excess": w1, "robust_sharpe": w2, "r_cubed": w3}
          - quantiles: 各指标的 p1/p5/p10/p25/p50/p75/p90/p95/p99/min/max/mean/std
          - correlations: 指标间 Pearson 相关系数
          - scoring_params: 绝对评分的线性映射参数
          - metadata: 采样元信息
    """
    if param_space is None:
        param_space = DEFAULT_PARAM_SPACE
    if n_workers is None:
        n_workers = max(1, multiprocessing.cpu_count() - 1)

    print(f"\n{'='*60}")
    print(f"🔬 评分体系校准 — 全域随机采样")
    print(f"{'='*60}")
    print(f"  参数空间:")
    for pname, (lo, hi) in param_space.items():
        print(f"    {pname:<22} [{lo}, {hi}]")
    print(f"  目标采样: {n_samples} 组  seed={seed}")

    # ── Step 1: LHS 采样 ──
    t0 = time.time()
    samples = latin_hypercube_sample(param_space, n_samples, seed=seed)
    print(f"  有效采样: {len(samples)} 组 (LHS + 拒绝采样)")

    # ── Step 2: 并行全量回测 ──
    batch_size = max(1, len(samples) // n_workers)
    batches = []
    for i in range(0, len(samples), batch_size):
        batches.append(samples[i:i + batch_size])

    all_metrics = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_worker_calibrate,
                            (batch, base_kwargs, df_base, benchmark_return)): idx
            for idx, batch in enumerate(batches)
        }
        for future in as_completed(futures):
            batch_results = future.result()
            all_metrics.extend(batch_results)
            done = sum(1 for f in futures if f.done())
            print(f"    [{done}/{len(batches)} batches] 累计有效: {len(all_metrics)}")

    elapsed = time.time() - t0
    print(f"  有效回测结果: {len(all_metrics)} 条  耗时: {elapsed:.0f}s")

    if len(all_metrics) < 30:
        print("  ⚠️ 有效样本不足 (<30)，返回默认校准")
        return _fallback_calibration(param_space)

    # ── Step 3: 计算熵权 ──
    metrics_df = pd.DataFrame(all_metrics)
    weights = compute_weights(metrics_df,method="critic")

    # ── Step 4: 计算分位数阈值 ──
    quantiles = {}
    for metric in ["excess", "robust_sharpe", "r_cubed"]:
        vals = metrics_df[metric]
        quantiles[metric] = {
            "p1":  round(float(vals.quantile(0.01)), 4),
            "p5":  round(float(vals.quantile(0.05)), 4),
            "p10": round(float(vals.quantile(0.10)), 4),
            "p25": round(float(vals.quantile(0.25)), 4),
            "p50": round(float(vals.quantile(0.50)), 4),
            "p75": round(float(vals.quantile(0.75)), 4),
            "p90": round(float(vals.quantile(0.90)), 4),
            "p95": round(float(vals.quantile(0.95)), 4),
            "p99": round(float(vals.quantile(0.99)), 4),
            "min":  round(float(vals.min()), 4),
            "max":  round(float(vals.max()), 4),
            "mean": round(float(vals.mean()), 4),
            "std":  round(float(vals.std()), 4),
        }

    # ── Step 5: 计算绝对评分参数 ──
    # 使用 p5/p95 作为线性映射锚点：[p5 → 5分, p95 → 95分]
    # 超出范围的值截断到 [1, 99]
    scoring_params = {}
    for metric in ["excess", "robust_sharpe", "r_cubed"]:
        q = quantiles[metric]
        p5, p95 = q["p5"], q["p95"]
        if p95 - p5 > 1e-10:
            scoring_params[metric] = {
                "type": "linear_quantile",
                "low_val": p5,
                "high_val": p95,
                "low_score": 5.0,
                "high_score": 95.0,
                "slope": 90.0 / (p95 - p5),
                "intercept": 5.0 - 90.0 * p5 / (p95 - p5),
            }
        else:
            scoring_params[metric] = {
                "type": "constant",
                "value": 50.0,
            }

    # ── Step 6: 指标间相关性 ──
    corr_df = metrics_df[["excess", "robust_sharpe", "r_cubed"]].corr()
    correlations = {
        "excess_vs_robust_sharpe": round(float(corr_df.loc["excess", "robust_sharpe"]), 4),
        "excess_vs_r_cubed": round(float(corr_df.loc["excess", "r_cubed"]), 4),
        "robust_sharpe_vs_r_cubed": round(float(corr_df.loc["robust_sharpe", "r_cubed"]), 4),
    }

    calibration = {
        "weights": {k: round(v, 4) for k, v in weights.items()},
        "quantiles": quantiles,
        "scoring_params": scoring_params,
        "correlations": correlations,
        "metadata": {
            "n_samples": len(all_metrics),
            "n_attempted": len(samples),
            "param_space": {k: list(v) for k, v in param_space.items()},
            "ts_code": base_kwargs.get("ts_code", ""),
            "stock_name": base_kwargs.get("stock_name", ""),
            "benchmark_return": benchmark_return,
            "seed": seed,
            "calibrated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    }

    # ── 打印结果 ──
    print(f"\n  📊 校准结果:")
    print(f"  {'─'*50}")
    print(f"  熵权:  excess={weights['excess']:.3f}"
          f"  robust_sharpe={weights['robust_sharpe']:.3f}"
          f"  r_cubed={weights['r_cubed']:.3f}")
    print(f"  指标相关性:")
    print(f"    excess ↔ robust_sharpe: {correlations['excess_vs_robust_sharpe']:+.3f}")
    print(f"    excess ↔ r_cubed:       {correlations['excess_vs_r_cubed']:+.3f}")
    print(f"    robust_sharpe ↔ r_cubed: {correlations['robust_sharpe_vs_r_cubed']:+.3f}")
    print(f"  分位数阈值 (p5 → p95):")
    for metric in ["excess", "robust_sharpe", "r_cubed"]:
        q = quantiles[metric]
        print(f"    {metric:<16}  [{q['p5']:>8.2f}, {q['p95']:>8.2f}]  "
              f"P50={q['p50']:>8.2f}  σ={q['std']:>6.2f}")
    print(f"  绝对评分公式 (linear_quantile):")
    for metric in ["excess", "robust_sharpe", "r_cubed"]:
        sp = scoring_params[metric]
        if sp["type"] == "linear_quantile":
            print(f"    {metric:<16}  score = {sp['slope']:.2f} × val"
                  f" + {sp['intercept']:.2f}"
                  f"  (clamp to [1, 99])")

    return calibration


def _fallback_calibration(param_space: dict | None = None) -> dict:
    """样本不足时的默认校准值。"""
    sp = param_space or DEFAULT_PARAM_SPACE
    return {
        "weights": {"excess": 0.50, "robust_sharpe": 0.40, "r_cubed": 0.10},
        "quantiles": {},
        "scoring_params": {
            "excess": {"type": "constant", "value": 50.0},
            "robust_sharpe": {"type": "constant", "value": 50.0},
            "r_cubed": {"type": "constant", "value": 50.0},
        },
        "correlations": {},
        "metadata": {
            "n_samples": 0,
            "fallback": True,
            "param_space": {k: list(v) for k, v in sp.items()},
            "calibrated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    }


# ═══════════════════════════════════════════════════════════════
# JSON 持久化
# ═══════════════════════════════════════════════════════════════

def save_calibration(calibration: dict, filepath: str):
    """持久化校准结果到 JSON 文件。"""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(calibration, f, indent=2, ensure_ascii=False)
    print(f"  💾 校准结果已保存: {filepath}")


def load_calibration(filepath: str) -> dict:
    """从 JSON 文件加载校准结果。"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"校准文件不存在: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════
# CLI — 独立运行校准
# ═══════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="评分体系校准 — 全域随机采样 → 熵权 + 分位数阈值")
    parser.add_argument("--ts-code", default="09992.HK", help="股票代码")
    parser.add_argument("--n-samples", type=int, default=1000,
                        help="LHS 采样数量 (默认: 500)")
    parser.add_argument("--workers", type=int,
                        default=max(1, multiprocessing.cpu_count() - 1),
                        help="并行进程数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--output", default=None,
                        help="输出 JSON 路径 (默认: result/calibration_{ts_code}.json)")
    parser.add_argument("--param-space", default=None,
                        help="自定义参数空间 JSON 文件路径")
    args = parser.parse_args()

    # ── 激活配置 ──
    common_config.init_config(args.ts_code)

    # ── 加载自定义参数空间 ──
    param_space = None
    if args.param_space:
        with open(args.param_space, "r") as f:
            raw = json.load(f)
            param_space = {k: tuple(v) for k, v in raw.items()}

    # ── 加载数据 ──
    base_kwargs = copy.deepcopy(common_config.BASE_KWARGS)

    config0 = BacktestConfig(**base_kwargs)
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

    # 买入持有基准收益
    mask = df_base["Date"] >= common_config.BASE_KWARGS["pm_start_date"]
    close = df_base.loc[mask, "Close"]
    bm = round((close.iloc[-1] / close.iloc[0] - 1) * 100, 2) if len(close) >= 2 else 0.0
    print(f"  数据: {len(df_base)} 条, 买入持有基准: {bm:+.2f}%")

    # ── 校准 ──
    calibration = calibrate(
        base_kwargs, df_base, bm,
        param_space=param_space,
        n_samples=args.n_samples,
        n_workers=args.workers,
        seed=args.seed,
    )

    # ── 保存 ──
    output_path = args.output or os.path.join(
        PROJECT_ROOT, "result",
        f"calibration_{args.ts_code.replace('.', '_')}.json")
    save_calibration(calibration, output_path)

    # ── 自动更新共享评分配置 ──
    common_config.update_scoring_config(args.ts_code, calibration)

    print(f"\n✅ 校准完成！后续脚本可使用:")
    print(f"   --calibration-file {output_path}")


if __name__ == "__main__":
    main()
