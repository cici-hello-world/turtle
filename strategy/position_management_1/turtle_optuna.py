#!/usr/bin/env python3
"""海龟策略参数优化 — Optuna 贝叶斯优化 (TPE)

用 Optuna 在连续参数空间直接搜索最优参数组合，自动平衡
exploration/exploitation，不受预设网格约束。

三窗口综合目标：
  objective = 0.5 × 滚动中位数 + 0.3 × 模拟加权 + 0.2 × 全量得分

每项得分用 composite_score（绝对评分，非百分位排名），
确保不同搜索空间下同一参数组合得分不变。

用法：
  python strategy/position_management_1/turtle_optuna.py --trials 200 --workers 8
  python strategy/position_management_1/turtle_optuna.py --trials 50   # 快速粗扫
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
import optuna

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from core.config import BacktestConfig
from core.data import fetch_data
from core.indicators import compute_indicators
from core.scoring import entropy_weights, compute_percentile_scores, apply_reward_penalty

from strategy.position_management_1.turtle_optimize import (
    run_one_backtest, _constraint_ok, _map_stats_keys, composite_score,
)
from strategy.position_management_1.grid_search_pm import (
    prepare_scenario_dfs, SCENARIO_LABEL_MAP, get_scenario_probs,
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
    user_dmi=True, pm_adaptive=True, pm_adaptive_grid=True,
    pm_dynamic_position=False, ema_fast=0, ema_slow=0,
    pm_initial_shares=2, initial_capital=40000,
    data_source="tushare",
)
# BASE_KWARGS = dict(
#     ts_code="03690.HK",
#     stock_name="美团",
#     market="HK",
#     mode="position_management_1",
#     per_shares=200,
#     start_date="20181023",
#     pm_start_date="20180920",
#     user_dmi=True,
#     pm_adaptive=True,
#     pm_adaptive_grid=True,       # 打开网格自适应
#     pm_dynamic_position=False,
#     ema_fast=0,
#     ema_slow=0,
#     pm_initial_shares=2,
#     initial_capital=40000,
#     # pm_agg_k 由 SWEEP_RANGES 覆盖
#     data_source="tushare",
# )

# 参数搜索边界（连续空间）
PARAM_SPACE = {
    "pm_grid_down":      (0.05, 0.10),
    "pm_grid_up":        (0.03, 0.08),
    "pm_reduce_pct":     (0.50, 0.75),
    "pm_add_pct":        (0.30, 0.50),
    "pm_adx_threshold":  (25, 50),
    "pm_agg_k":          (0.4, 0.9),
}

# 滚动窗口参数
WINDOW_BARS, STEP_BARS, WARMUP = 252, 20, 60

# 综合权重 (滚动/模拟/全量)
W_ROLLING, W_SIM, W_FULL = 0.5, 0.3, 0.2

# ═══════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════

def _now():
    return time.strftime("%H:%M:%S")


def _split_batches(items: list, n_batches: int) -> list[list]:
    if n_batches <= 1 or len(items) <= n_batches:
        return [items]
    k, m = divmod(len(items), n_batches)
    batches, start = [], 0
    for i in range(n_batches):
        end = start + k + (1 if i < m else 0)
        if end > start:
            batches.append(items[start:end])
        start = end
    return batches


# ═══════════════════════════════════════════════════════════════
# Worker 函数（模块级别，供 ProcessPoolExecutor 使用）
# ═══════════════════════════════════════════════════════════════

def _worker_rolling(args):
    """对一批 (params, ws, df_win, bm) 运行滚动窗口回测，返回 scores。"""
    batch, base_kwargs = args
    scores = []
    for params, ws, df_win, bm in batch:
        win_kwargs = {**base_kwargs, "pm_start_date": ws}
        stats = run_one_backtest(params, win_kwargs, df_win, benchmark_return=bm)
        if stats and stats.get("总交易", 0) > 0:
            scores.append(composite_score(stats))
    return scores


def _worker_scenario(args):
    """对一批 (params, sc_label, sc_df, sc_bm) 运行场景回测，返回 [(label, prob, score), ...]"""
    batch, base_kwargs, sim_start = args
    results = []
    for params, sc_label, sc_df, sc_bm in batch:
        sc_kwargs = {**base_kwargs, "pm_start_date": sim_start}
        stats = run_one_backtest(params, sc_kwargs, sc_df, benchmark_return=sc_bm)
        if stats and stats.get("总交易", 0) > 0:
            prob = get_scenario_probs(BASE_KWARGS["ts_code"]).get(sc_label, 0.0)
            results.append((prob, composite_score(stats)))
    return results


# ═══════════════════════════════════════════════════════════════
# 单次参数评估（绝对评分，不依赖搜索空间）
# ═══════════════════════════════════════════════════════════════

class ParamEvaluator:
    """参数评估器 — 持有共享数据，对单个参数组合做三窗口评分。"""

    def __init__(self, base_kwargs, df_base, scenario_dfs, scenario_benchmarks,
                 sim_start, full_bm, n_workers):
        self.base_kwargs = base_kwargs
        self.df_base = df_base
        self.scenario_dfs = scenario_dfs
        self.scenario_benchmarks = scenario_benchmarks
        self.sim_start = sim_start
        self.full_bm = full_bm
        self.n_workers = n_workers

        # 预计算滚动窗口参数
        dates = pd.to_datetime(df_base["Date"])
        self._n = len(df_base)
        self._dates = dates
        self._rolling_windows = []
        for start_idx in range(WARMUP, self._n - WINDOW_BARS, STEP_BARS):
            end_idx = start_idx + WINDOW_BARS
            warmup_start = max(0, start_idx - WARMUP)
            df_win = df_base.iloc[warmup_start:end_idx]
            ws = pd.Timestamp(dates.iloc[start_idx]).strftime("%Y%m%d")
            win_mask = df_win["Date"] >= ws
            win_close = df_win.loc[win_mask, "Close"]
            bm = ((win_close.iloc[-1] / win_close.iloc[0] - 1) * 100
                  if len(win_close) >= 2 else 0.0)
            self._rolling_windows.append((ws, df_win, bm))

    def evaluate(self, params: dict) -> dict | None:
        """运行三窗口评估，返回 {combined, full_score, rolling_score,
        sim_score, stats_full, n_rolling, n_sim, ...} 或 None。"""
        # ── Stage A: 全量历史（便宜，先跑）──
        stats_full = run_one_backtest(params, self.base_kwargs,
                                      self.df_base, benchmark_return=self.full_bm)
        if not stats_full:
            return None
        full_score = composite_score(stats_full)

        # ── Stage B: 滚动窗口（昂贵，并行）──
        rolling_scores = self._run_rolling_parallel(params)

        if len(rolling_scores) >= 5:
            rolling_score = float(np.median(rolling_scores))
        elif rolling_scores:
            rolling_score = float(np.mean(rolling_scores))
        else:
            rolling_score = 0.0

        # ── Stage C: 8种场景模拟（并行）──
        sim_score = self._run_scenario_parallel(params)

        # ── 综合 ──
        combined = (W_ROLLING * rolling_score +
                    W_SIM * sim_score +
                    W_FULL * full_score)

        return {
            "combined": round(combined, 2),
            "full_score": round(full_score, 2),
            "rolling_score": round(rolling_score, 2),
            "sim_score": round(sim_score, 2),
            "stats_full": stats_full,
            "n_rolling": len(rolling_scores),
            "n_sim": sum(1 for _ in SCENARIO_LABEL_MAP),
        }

    def _run_rolling_parallel(self, params: dict) -> list[float]:
        """并行运行所有滚动窗口回测。"""
        if not self._rolling_windows:
            return []

        # 构建任务列表
        tasks = [(params, ws, df_win, bm)
                 for ws, df_win, bm in self._rolling_windows]
        batches = _split_batches(tasks, self.n_workers)

        all_scores = []
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = [executor.submit(_worker_rolling, (b, self.base_kwargs))
                       for b in batches]
            for future in as_completed(futures):
                all_scores.extend(future.result())
        return all_scores

    def _run_scenario_parallel(self, params: dict) -> float:
        """并行运行8种场景回测，返回概率加权得分。"""
        tasks = [(params, sc_label, self.scenario_dfs[sc_label],
                  self.scenario_benchmarks[sc_label])
                 for sc_label in SCENARIO_LABEL_MAP]

        all_pairs = []
        with ProcessPoolExecutor(max_workers=min(self.n_workers, len(tasks))) as executor:
            futures = [executor.submit(_worker_scenario,
                                       ([t], self.base_kwargs, self.sim_start))
                       for t in tasks]
            for future in as_completed(futures):
                all_pairs.extend(future.result())

        if not all_pairs:
            return 0.0
        total_prob = sum(p for p, _ in all_pairs)
        if total_prob <= 0:
            return 0.0
        return sum(p * s for p, s in all_pairs) / total_prob


# ═══════════════════════════════════════════════════════════════
# Optuna 目标函数
# ═══════════════════════════════════════════════════════════════

class OptunaObjective:
    """Optuna 目标函数（可调用对象）。"""

    def __init__(self, evaluator: ParamEvaluator):
        self.evaluator = evaluator
        self.trial_count = 0
        self.best_score = float('-inf')
        self.best_params = None

    def __call__(self, trial: optuna.Trial) -> float:
        self.trial_count += 1

        # ── 采样参数（依赖采样天然满足约束）──
        gu = trial.suggest_float("pm_grid_up",
                                 PARAM_SPACE["pm_grid_up"][0],
                                 PARAM_SPACE["pm_grid_up"][1])
        gd = trial.suggest_float("pm_grid_down",
                                 gu + 0.005,
                                 PARAM_SPACE["pm_grid_down"][1])

        ap = trial.suggest_float("pm_add_pct",
                                 PARAM_SPACE["pm_add_pct"][0],
                                 PARAM_SPACE["pm_add_pct"][1])
        rp = trial.suggest_float("pm_reduce_pct",
                                 ap + 0.05,
                                 PARAM_SPACE["pm_reduce_pct"][1])

        adx = trial.suggest_int("pm_adx_threshold",
                                PARAM_SPACE["pm_adx_threshold"][0],
                                PARAM_SPACE["pm_adx_threshold"][1])
        k = trial.suggest_float("pm_agg_k",
                                PARAM_SPACE["pm_agg_k"][0],
                                PARAM_SPACE["pm_agg_k"][1])

        params = dict(
            pm_grid_down=round(gd, 3),
            pm_grid_up=round(gu, 3),
            pm_reduce_pct=round(rp, 2),
            pm_add_pct=round(ap, 2),
            pm_adx_threshold=adx,
            pm_agg_k=round(k, 3),
        )

        # ── Stage A: 先跑全量历史（便宜）──
        stats_full = run_one_backtest(params, self.evaluator.base_kwargs,
                                      self.evaluator.df_base,
                                      benchmark_return=self.evaluator.full_bm)
        if not stats_full:
            return float('-inf')

        full_score = composite_score(stats_full)
        trial.report(full_score, step=0)

        # 用 MedianPruner 在 Stage A 后剪枝（TPE 会学习避开低分区）
        if trial.should_prune():
            raise optuna.TrialPruned()

        # ── Stage B + C: 完整三窗口评估 ──
        result = self.evaluator.evaluate(params)
        if result is None:
            return float('-inf')

        # 记录用户属性
        trial.set_user_attr("full_score", result["full_score"])
        trial.set_user_attr("rolling_score", result["rolling_score"])
        trial.set_user_attr("sim_score", result["sim_score"])
        trial.set_user_attr("n_rolling", result["n_rolling"])
        trial.set_user_attr("CAGR", round(result["stats_full"].get("CAGR (%)", 0), 2))
        trial.set_user_attr("MaxDD", round(result["stats_full"].get("最大回撤 (%)", 0), 2))
        trial.set_user_attr("excess", round(result["stats_full"].get("excess",
                            result["stats_full"].get("超额收益 (%)", 0) or 0), 2))
        trial.set_user_attr("trades", result["stats_full"].get("总交易", 0))

        combined = result["combined"]

        # 跟踪全局最优
        if combined > self.best_score:
            self.best_score = combined
            self.best_params = params

        # 进度
        if self.trial_count % 10 == 0:
            print(f"    [{_now()}] trial {self.trial_count}: "
                  f"combined={combined:.1f}  "
                  f"full={result['full_score']:.1f}  "
                  f"rolling={result['rolling_score']:.1f}  "
                  f"sim={result['sim_score']:.1f}  "
                  f"(best={self.best_score:.1f})")

        return combined


# ═══════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="海龟策略 Optuna 贝叶斯优化")
    parser.add_argument("--trials", type=int, default=100,
                        help="Optuna 试验次数 (默认: 100)")
    parser.add_argument("--workers", type=int,
                        default=max(1, multiprocessing.cpu_count() - 1),
                        help="内部并行工作进程数（每个 trial 内并行 Stages）")
    parser.add_argument("--no-prune", action="store_true",
                        help="关闭 MedianPruner 剪枝")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (默认: 42)")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════╗")
    print("║  海龟策略 Optuna 贝叶斯优化 (TPE)                       ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"\n  股票: {BASE_KWARGS['stock_name']} ({BASE_KWARGS['ts_code']})")
    print(f"  参数空间: { {k: v for k, v in PARAM_SPACE.items()} }")
    print(f"  综合权重: rolling={W_ROLLING}  sim={W_SIM}  full={W_FULL}")
    print(f"  试验次数: {args.trials}  内部并行: {args.workers} workers")
    print(f"  剪枝: {'关闭' if args.no_prune else 'MedianPruner (Stage A 后)'}")
    print()

    # ── 加载数据 ──
    print(f"[{_now()}] 📥 加载数据 ...")
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

    # 基准收益
    mask = df_base["Date"] >= BASE_KWARGS["pm_start_date"]
    close = df_base.loc[mask, "Close"]
    full_bm = round((close.iloc[-1] / close.iloc[0] - 1) * 100, 2) if len(close) >= 2 else 0.0
    print(f"   数据: {len(df_base)} 条, 买入持有基准: {full_bm:+.2f}%")

    n = len(df_base)
    n_windows = max(0, (n - WARMUP - WINDOW_BARS) // STEP_BARS + 1)
    print(f"   滚动窗口: {WINDOW_BARS}d × 步长{STEP_BARS}d → ~{n_windows} 个窗口")

    # ── 准备场景数据 ──
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

    # ── 构建评估器和目标函数 ──
    evaluator = ParamEvaluator(
        BASE_KWARGS, df_base, scenario_dfs, scenario_benchmarks,
        sim_start, full_bm, args.workers,
    )
    objective_fn = OptunaObjective(evaluator)

    # ── 创建 Optuna Study ──
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=30, n_warmup_steps=0, interval_steps=1,
    ) if not args.no_prune else optuna.pruners.NopPruner()

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=args.seed,
            n_startup_trials=30,  # 前30次随机探索
            multivariate=True,     # 考虑参数间相关性
        ),
        pruner=pruner,
        study_name="turtle_pm1_optuna",
    )

    # ── 运行优化 ──
    print(f"\n{'='*60}")
    print(f"[{_now()}] 🔍 开始 Optuna 优化 ({args.trials} trials)")
    print(f"{'='*60}\n")

    t0 = time.time()
    study.optimize(objective_fn, n_trials=args.trials, n_jobs=1)
    elapsed = time.time() - t0

    # ── 结果 ──
    print(f"\n{'='*60}")
    print(f"🏆 Optuna 优化完成")
    print(f"{'='*60}")
    print(f"  耗时: {elapsed/60:.1f} min  "
          f"完成: {len(study.trials)}/{args.trials}  "
          f"剪枝: {sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)}")

    best = study.best_trial
    print(f"\n  🥇 最优参数 (trial #{best.number}):")
    for pname in ["pm_grid_down", "pm_grid_up", "pm_reduce_pct",
                   "pm_add_pct", "pm_adx_threshold", "pm_agg_k"]:
        val = best.params[pname]
        if pname == "pm_adx_threshold":
            print(f"     {pname:<22} = {int(val)}")
        else:
            print(f"     {pname:<22} = {val:.3f}")
    print(f"     ──────────────────────")
    print(f"     综合得分 (absolute):  {best.value:.2f}")
    print(f"     全量得分:             {best.user_attrs.get('full_score', 'N/A')}")
    print(f"     滚动得分:             {best.user_attrs.get('rolling_score', 'N/A')}")
    print(f"     模拟得分:             {best.user_attrs.get('sim_score', 'N/A')}")
    print(f"     CAGR:                 {best.user_attrs.get('CAGR', 'N/A')}%")
    print(f"     最大回撤:             {best.user_attrs.get('MaxDD', 'N/A')}%")

    # ── 参数重要性 ──
    print(f"\n  📊 参数重要性 (TPE 增益):")
    try:
        importances = optuna.importance.get_param_importances(study)
        for pname, imp in importances.items():
            bar = "█" * int(imp * 40)
            print(f"     {pname:<22} {imp:.3f}  {bar}")
    except Exception as e:
        print(f"     ⚠️ 无法计算: {e}")

    # ── 保存 ──
    out_dir = os.path.join(PROJECT_ROOT, "result")
    os.makedirs(out_dir, exist_ok=True)

    # Excel: 所有试验记录
    trial_rows = []
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE:
            trial_rows.append({
                "trial": t.number,
                "value": t.value,
                **{p: t.params.get(p, None) for p in PARAM_SPACE},
                **t.user_attrs,
            })
    if trial_rows:
        xlsx_path = os.path.join(out_dir, "turtle_optuna_result.xlsx")
        df_trials = pd.DataFrame(trial_rows)
        df_trials.sort_values("value", ascending=False, inplace=True)
        df_trials.to_excel(xlsx_path, sheet_name="Optuna结果", index=False)
        print(f"\n📁 试验记录: {xlsx_path} ({len(trial_rows)} 条完成)")

    # 参数重要性图 (HTML，浏览器打开即可查看)
    try:
        fig = optuna.visualization.plot_param_importances(study)
        html_path = os.path.join(out_dir, "turtle_optuna_importance.html")
        fig.write_html(html_path)
        print(f"📁 参数重要性: result/turtle_optuna_importance.html")
    except Exception as e:
        print(f"     ⚠️ 参数重要性图失败: {e}")

    # 优化历史图 (HTML)
    try:
        fig = optuna.visualization.plot_optimization_history(study)
        html_path = os.path.join(out_dir, "turtle_optuna_history.html")
        fig.write_html(html_path)
        print(f"📁 优化历史: result/turtle_optuna_history.html")
    except Exception as e:
        print(f"     ⚠️ 优化历史图失败: {e}")

    # ── 采样分布报告：帮助判断下次 seed 或 PARAM_SPACE 如何调整 ──
    if trial_rows:
        df = pd.DataFrame(trial_rows)
        top10 = df.nlargest(10, "value")

        print(f"\n{'='*70}")
        print(f"📊 采样密集区域分析（用于调优下次搜索范围）")
        print(f"{'='*70}")
        print(f"  {'参数':<22} {'搜索边界':>18} {'全部采样80%区间':>20} {'TOP10密集区':>20} {'覆盖判断'}")
        print(f"  {'─'*90}")

        for pname in ["pm_grid_down", "pm_grid_up", "pm_reduce_pct",
                       "pm_add_pct", "pm_adx_threshold", "pm_agg_k"]:
            lo, hi = PARAM_SPACE[pname]
            vals = df[pname].dropna()
            if len(vals) < 3:
                continue

            p10, p90 = float(np.percentile(vals, 10)), float(np.percentile(vals, 90))
            top_p25 = float(np.percentile(top10[pname], 25))
            top_p75 = float(np.percentile(top10[pname], 75))

            # 判断边界是否够用
            range_used = p90 - p10
            range_total = hi - lo
            utilization = range_used / range_total if range_total > 0 else 0

            left_hit = (p10 - lo) / range_total < 0.05 if range_total > 0 else False
            right_hit = (hi - p90) / range_total < 0.05 if range_total > 0 else False

            if left_hit and right_hit:
                verdict = "🔒 两端触边 → 扩大范围"
            elif left_hit:
                verdict = "⬅️ 低端触边 → 向下扩展"
            elif right_hit:
                verdict = "➡️ 高端触边 → 向上扩展"
            elif utilization > 0.5:
                verdict = "✅ 分布均匀 → 可收缩"
            else:
                verdict = "✅ 充分覆盖"

            print(f"  {pname:<22} [{lo:.3f}, {hi:.2f}] "
                  f"[{p10:.3f}, {p90:.3f}]  "
                  f"[{top_p25:.3f}, {top_p75:.3f}]  "
                  f"{verdict}")

        print(f"\n  解读：全部采样80%区间 = 大部分 trial 落在哪里")
        print(f"        TOP10密集区 = 最优10个 trial 的参数聚集在哪个范围")
        print(f"        如果最优值靠近搜索边界，下次应向外扩展该边界")
        print(f"        如果最优值远离边界且集中，下次可收缩范围以提升精度")

    print(f"\n[{_now()}] ✅ Optuna 优化完成！")


if __name__ == "__main__":
    main()
