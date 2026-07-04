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
from datetime import timedelta

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
from core.scoring import entropy_weights, compute_percentile_scores, apply_reward_penalty

# 复用已有函数
from strategy.position_management_1.turtle_optimize import (
    run_one_backtest, _constraint_ok, _map_stats_keys,
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
    ts_code="09992.HK",
    stock_name="泡泡玛特",
    market="HK",
    mode="position_management_1",
    per_shares=200,
    start_date="20201211",
    pm_start_date="20210112",
    user_dmi=True,
    pm_adaptive=True,
    pm_adaptive_grid=False,      # 关闭网格自适应
    pm_dynamic_position=False,
    ema_fast=0,
    ema_slow=0,
    pm_initial_shares=2,
    initial_capital=40000,
    pm_agg_k=1.0,                # 屏蔽
    pm_take_profit_ratio=1.30,
    data_source="tushare",
)

# 5个核心参数 — 基于数据驱动选择
SWEEP_RANGES = {
    "pm_grid_down":      [0.05, 0.07, 0.10, 0.12],
    "pm_grid_up":        [0.03, 0.04, 0.05, 0.07],
    "pm_reduce_pct":     [0.35, 0.50, 0.60, 0.75],
    "pm_add_pct":        [0.25, 0.30, 0.35, 0.40, 0.50],
    "pm_adx_threshold":  [25, 30, 35],
}

PARAM_LABELS = {
    "pm_grid_down": "gd",
    "pm_grid_up": "gu",
    "pm_reduce_pct": "rp",
    "pm_add_pct": "ap",
    "pm_adx_threshold": "adx",
}

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
    print(f"[{_now()}] 🅰 Stage A: 全量历史回测 ({len(param_combos)} 组)")
    print(f"{'='*60}")

    all_results = []  # 所有窗口×所有参数的结果
    t0 = time.time()

    for idx, params in enumerate(param_combos):
        stats = run_one_backtest(params, BASE_KWARGS, df_base, benchmark_return=full_bm)
        if stats:
            stats["_window"] = "full"
            stats["_params"] = params
            all_results.append(stats)

        if (idx + 1) % 100 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (idx + 1) * (len(param_combos) - idx - 1)
            print(f"    [{idx+1}/{len(param_combos)}] {elapsed:.0f}s ETA {eta/60:.0f}min  "
                  f"有效: {len(all_results)}")

    full_count = len(all_results)
    print(f"  全量完成: {full_count} 组有效  "
          f"({len(param_combos)-full_count} 组无交易)  "
          f"耗时 {(time.time()-t0)/60:.1f}min")

    # ═══════════════════════════════════════════════════════
    # Stage B: 真实滚动窗口回测
    # ═══════════════════════════════════════════════════════
    WINDOW_BARS = 252
    STEP_BARS = 20
    WARMUP = 60

    dates = pd.to_datetime(df_base["Date"])
    n = len(df_base)
    n_windows = max(0, (n - WARMUP - WINDOW_BARS) // STEP_BARS + 1)

    print(f"\n{'='*60}")
    print(f"[{_now()}] 🅱 Stage B: 滚动窗口回测 "
          f"({len(param_combos)}组 × ~{n_windows}窗口)")
    print(f"{'='*60}")
    print(f"  窗口: {WINDOW_BARS}天 × 步长{STEP_BARS}天 × 预热{WARMUP}天")

    t0 = time.time()
    rolling_count = 0

    for p_idx, params in enumerate(param_combos):
        for start_idx in range(WARMUP, n - WINDOW_BARS, STEP_BARS):
            end_idx = start_idx + WINDOW_BARS
            warmup_start = max(0, start_idx - WARMUP)
            df_win = df_base.iloc[warmup_start:end_idx].copy()
            ws = pd.Timestamp(dates.iloc[start_idx]).strftime("%Y%m%d")

            win_mask = df_win["Date"] >= ws
            win_close = df_win.loc[win_mask, "Close"]
            bm = ((win_close.iloc[-1] / win_close.iloc[0] - 1) * 100
                  if len(win_close) >= 2 else 0.0)

            win_kwargs = {**BASE_KWARGS, "pm_start_date": ws}
            stats = run_one_backtest(params, win_kwargs, df_win, benchmark_return=bm)
            if stats and stats.get("总交易", 0) > 0:
                stats["_window"] = f"rolling_{ws}"
                stats["_params"] = params
                all_results.append(stats)
                rolling_count += 1

        if (p_idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (p_idx + 1) * (len(param_combos) - p_idx - 1)
            print(f"    [{p_idx+1}/{len(param_combos)}] {elapsed:.0f}s ETA {eta/60:.0f}min  "
                  f"累计: {len(all_results)}")

    print(f"  滚动完成: +{rolling_count} 条  (总累计 {len(all_results)})  "
          f"耗时 {(time.time()-t0)/60:.1f}min")

    # ═══════════════════════════════════════════════════════
    # Stage C: 8种场景模拟回测
    # ═══════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"[{_now()}] 🅲 Stage C: 8种场景模拟回测 ({len(param_combos)}组 × 8场景)")
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
    for p_idx, params in enumerate(param_combos):
        for sc_label in SCENARIO_LABEL_MAP:
            sc_kwargs = {**BASE_KWARGS, "pm_start_date": sim_start}
            stats = run_one_backtest(params, sc_kwargs,
                                     scenario_dfs[sc_label],
                                     benchmark_return=scenario_benchmarks[sc_label])
            if stats and stats.get("总交易", 0) > 0:
                stats["_window"] = f"sim_{sc_label}"
                stats["_params"] = params
                all_results.append(stats)
                sim_count += 1

        if (p_idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (p_idx + 1) * (len(param_combos) - p_idx - 1)
            print(f"    [{p_idx+1}/{len(param_combos)}] {elapsed:.0f}s ETA {eta/60:.0f}min  "
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

    # Step 2: 从全量窗口子集计算熵权
    full_window_results = [r for r in all_results if r["_window"] == "full"]
    metrics_df = pd.DataFrame([
        {"excess": r["超额收益 (%)"], "robust_sharpe": r["稳健夏普比率"],
         "r_cubed": r["R立方"]}
        for r in full_window_results
    ])
    weights = entropy_weights(metrics_df)
    print(f"  熵权法权重: excess={weights['excess']:.3f}  "
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
                prob = SCENARIO_PROBS.get(sc_label, 0.0)
                if prob > 0:
                    sim_scores.append(r["score"] * prob)
                    sim_probs.append(prob)

        if sim_scores and sum(sim_probs) > 0:
            sim_weighted = sum(sim_scores) / sum(sim_probs)
        else:
            sim_weighted = 50.0

        # 综合 = 0.5 × 滚动中位数 + 0.3 × 模拟加权 + 0.2 × 全量
        combined = 0.5 * rolling_median + 0.3 * sim_weighted + 0.2 * full_score

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
    print(f"\n  🥇 最优参数组合:")
    print(f"     pm_grid_down       = {best['pm_grid_down']:.2f}")
    print(f"     pm_grid_up         = {best['pm_grid_up']:.2f}")
    print(f"     pm_reduce_pct      = {best['pm_reduce_pct']:.2f}")
    print(f"     pm_add_pct         = {best['pm_add_pct']:.2f}")
    print(f"     pm_adx_threshold   = {int(best['pm_adx_threshold'])}")
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

    xlsx_path = os.path.join(out_dir, "turtle_scoring_result.xlsx")
    df_summary = pd.DataFrame(summary)
    df_summary.to_excel(xlsx_path, sheet_name="综合排名", index=False)
    print(f"\n📁 综合排名: {xlsx_path} ({len(summary)} 组)")

    # 每窗口明细
    detail_path = os.path.join(out_dir, "turtle_scoring_detail.xlsx")
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
    main()
