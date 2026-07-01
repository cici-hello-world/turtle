#!/usr/bin/env python3
"""PM1 自适应网格策略 —— 四轮优化参数搜索

agg 参数由引擎根据 pm_agg_k 自动派生:
  gda = gd × k    gua = gu × k    rpa = min(rp + (1-k)×0.5, 0.95)

第1轮（粗筛）：逻辑约束 + 基本质量过滤
第2轮（排名）：R³ 复合评分排名（不过滤，仅展示）
第3轮（验证）：全量 8种模拟行情压力测试 → TOP 10
第4轮（鲁棒）：参数 ±15% 扰动敏感性检验 → 最终 2-3 组候选

用法：
  python grid_search_pm.py
"""

import itertools
import sys
import os
import time
import copy
from datetime import timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.config import BacktestConfig
from core.data import fetch_data
from core.indicators import compute_indicators
from core.engine import run_backtest
from core.statistics import calc_stats


# ═══════════════════════════════════════════════════════════════
# 评分函数
# ═══════════════════════════════════════════════════════════════

def composite_score(stats: dict) -> float:
    """个体评分（用于场景验证和敏感性检验）。

    超额40% + 稳健夏普40% + 总收益10% + R³10%。
    """
    r3 = min(stats.get("R立方", 0.0) or 0.0, 10.0)
    rs = stats.get("稳健夏普比率", 0.0) or 0.0
    excess = stats.get("超额收益 (%)", 0.0) or 0.0

    return (excess + 20) / 60 * 50 + (rs + 2) / 8 * 40 + r3 / 10 * 10


def compute_percentile_scores(results: list[dict]) -> list[dict]:
    """批量百分位归一化评分（用于第2轮排名）。

    四项指标百分位加权，超额收益为主，完全免疫极端值。
    """
    if len(results) < 3:
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


def rolling_window_cv(params: dict, base_kwargs: dict, df_base: pd.DataFrame,
                      window_bars: int = 252, step_bars: int = 63,
                      warmup_bars: int = 126) -> float:
    """滚动窗口交叉验证 —— 在真实数据的多个子区间上回测，取 P10 得分。

    惩罚只在特定时段表现好、其他时段崩盘的参数。
    """
    dates = df_base["Date"].values
    n = len(df_base)

    scores = []
    for start_idx in range(warmup_bars, n - window_bars, step_bars):
        end_idx = start_idx + window_bars
        warmup_start_idx = max(0, start_idx - warmup_bars)

        df_window = df_base.iloc[warmup_start_idx:end_idx].copy()
        ws = pd.Timestamp(dates[start_idx]).strftime("%Y%m%d")
        we = pd.Timestamp(dates[end_idx - 1]).strftime("%Y%m%d")

        window_kwargs = {**base_kwargs,
                         "pm_start_date": ws,
                         "start_date": pd.Timestamp(dates[warmup_start_idx]).strftime("%Y%m%d"),
                         "end_date": we}

        try:
            config_w = BacktestConfig(**{**window_kwargs, **params})
            trades, equity, eq_dates = run_backtest(df_window, config_w)

            # 该窗口的买入持有基准收益
            win_mask = df_window["Date"] >= ws
            win_close = df_window.loc[win_mask, "Close"]
            bm = ((win_close.iloc[-1] / win_close.iloc[0] - 1) * 100
                  if len(win_close) >= 2 else 0.0)

            stats = calc_stats(trades, config_w.initial_capital, equity, eq_dates,
                               benchmark_return=bm)
            scores.append(composite_score(stats))
        except Exception:
            scores.append(float("-inf"))

    valid = [s for s in scores if s > float("-inf")]
    if len(valid) < 3:
        return 0.0
    # P10: 最差的那10%窗口的得分上界 —— 低于此的窗口拖累直接反映在分数里
    return float(np.percentile(valid, 10))


# ═══════════════════════════════════════════════════════════════
# agg 参数派生 (与 engine._run_position_management_1 保持一致)
# ═══════════════════════════════════════════════════════════════

def derive_agg(gd, gu, rp, k):
    """由趋势参数 + 激进倍数 k 派生震荡市参数。

    k ∈ (0, 1]:  越小越激进
      k=0.4 → gda=gd×0.4  gua=gu×0.4  rpa=rp+0.30
      k=1.0 → gda=gd       gua=gu       rpa=rp
    """
    gda = round(gd * k, 3)
    gua = round(gu * k, 3)
    rpa = min(round(rp + (1 - k) * 0.5, 2), 0.95)
    return gda, gua, rpa


# ═══════════════════════════════════════════════════════════════
# 第3轮: 场景验证 — 预生成场景数据
# ═══════════════════════════════════════════════════════════════

# 场景概率 (来自 scenario_analysis.py 宏观估算)
SCENARIO_PROBS = {
    "缓慢阴跌": 0.35,
    "震荡向下": 0.20,
    "横盘震荡": 0.15,
    "V形反转":  0.10,
    "震荡向上": 0.10,
    "单边下跌": 0.05,
    "倒V反转":  0.03,
    "单边上涨": 0.02,
}

SCENARIO_LABEL_MAP = {
    "单边上涨":     "up",
    "单边下跌":     "down",
    "震荡向上":     "volatile_up",
    "震荡向下":     "volatile_down",
    "横盘震荡":     "sideways",
    "V形反转":      "v_reversal",
    "倒V反转":      "inv_v_reversal",
    "缓慢阴跌":     "slow_grind_down",
}


def prepare_scenario_dfs(last_price: float, last_date,
                         config0: BacktestConfig,
                         df_base: pd.DataFrame,
                         trading_days: int = 252) -> tuple[dict, dict]:
    """生成8种模拟行情数据并预计算指标。

    模拟从 2025-10-01 到 last_date 的 K 线，与真实回测区间对齐。

    返回 (scenario_dfs, benchmark_returns)
    - scenario_dfs: {中文标签: df_with_indicators}
    - benchmark_returns: {中文标签: 基准收益率%}
    """
    from simulate_market_scenarios import generate_ohlcv
    from core.cache import load as cache_load

    cached = cache_load(config0.ts_code, config0.market)
    # 模拟从最后真实日期之后开始，代表未来未知行情
    start = (last_date + timedelta(days=1)).strftime("%Y%m%d")
    trading_days = 252

    scenario_dfs = {}
    benchmark_returns = {}
    for label, trend_type in SCENARIO_LABEL_MAP.items():
        df_sim = generate_ohlcv(start, trading_days, last_price, trend_type)

        sim_start = df_sim["Date"].min()
        if cached is not None and not cached.empty:
            hist = cached[cached["Date"] < sim_start].copy()
            if not hist.empty:
                common_cols = df_sim.columns.intersection(hist.columns)
                df_sim = pd.concat([hist[common_cols], df_sim[common_cols]],
                                   ignore_index=True)
                df_sim = df_sim.sort_values("Date").reset_index(drop=True)

        df_sim = compute_indicators(
            df_sim,
            entry_period=config0.entry_period,
            exit_period=config0.exit_period,
            atr_period=config0.atr_period,
            ema_fast=config0.ema_fast,
            ema_slow=config0.ema_slow,
            channel_type=config0.channel_type,
            bb_period=config0.bb_period,
            bb_k=config0.bb_k,
            am_ma_period=config0.am_ma_period,
            mode=config0.mode,
        )
        scenario_dfs[label] = df_sim

        # 计算该场景的买入持有基准收益
        sim_mask = df_sim["Date"] >= sim_start
        sim_close = df_sim.loc[sim_mask, "Close"]
        sp = float(sim_close.iloc[0])
        ep = float(sim_close.iloc[-1])
        bm = (ep - sp) / sp * 100
        benchmark_returns[label] = bm
        print(f"     ✓ {label}: ¥{sp:.1f}→¥{ep:.1f} ({bm:+.1f}%)")

    return scenario_dfs, benchmark_returns


# ═══════════════════════════════════════════════════════════════
# 第4轮: 参数敏感性检验
# ═══════════════════════════════════════════════════════════════

def _compute_combined_score(params: dict, base_kwargs: dict,
                            df_base: pd.DataFrame,
                            benchmark_return: float,
                            scenario_dfs: dict | None,
                            scenario_benchmarks: dict | None,
                            sim_start_date: str | None) -> float:
    """计算真实+模拟的综合得分。

    综合得分 = 真实数据 composite_score × 0.3 + 模拟概率加权 composite_score × 0.7
    若未传入模拟数据，退化为仅真实数据得分。
    """
    # ── 真实数据得分 ──
    config_real = BacktestConfig(**{**base_kwargs, **params})
    trades, equity, eq_dates = run_backtest(df_base.copy(), config_real)
    stats_real = calc_stats(trades, config_real.initial_capital, equity, eq_dates,
                            benchmark_return=benchmark_return)
    real_score = composite_score(stats_real)

    if not scenario_dfs:
        return real_score

    # ── 模拟行情得分（概率加权）──
    scenario_scores = []
    probs = []
    for label in SCENARIO_LABEL_MAP:
        if label not in scenario_dfs:
            continue
        scenario_kwargs = {**base_kwargs, "pm_start_date": sim_start_date}
        config_s = BacktestConfig(**{**scenario_kwargs, **params})
        try:
            trades_s, equity_s, eq_dates_s = run_backtest(
                scenario_dfs[label].copy(), config_s)
            stats_s = calc_stats(trades_s, config_s.initial_capital,
                                 equity_s, eq_dates_s,
                                 benchmark_return=scenario_benchmarks.get(label, 0.0))
            scenario_scores.append(composite_score(stats_s))
            probs.append(SCENARIO_PROBS.get(label, 0.0))
        except Exception:
            scenario_scores.append(0.0)
            probs.append(SCENARIO_PROBS.get(label, 0.0))

    total_prob = sum(probs)
    if total_prob > 0:
        weighted_score = sum(s * p / total_prob for s, p in zip(scenario_scores, probs))
    else:
        weighted_score = 0.0

    return real_score * 0.3 + weighted_score * 0.7


def sensitivity_test(params: dict, base_kwargs: dict, df_base: pd.DataFrame,
                     benchmark_return: float = 0.0,
                     perturb_pct: float = 0.15,
                     scenario_dfs: dict | None = None,
                     scenario_benchmarks: dict | None = None,
                     sim_start_date: str | None = None) -> dict:
    """对一组参数做 ±perturb_pct 扰动，评估得分稳定性。

    同时在真实数据和8种模拟行情上计算扰动前后的综合得分变化。
    综合得分 = 真实得分 × 0.3 + 模拟概率加权得分 × 0.7

    k 的扰动会让所有 agg 参数联动变化。

    返回:
      base_score:        基准综合得分
      avg_perturbed:     扰动后综合得分均值
      decay_pct:         综合得分衰减率 (%)
      base_score_real:   基准真实数据得分（仅当同时有模拟数据时返回）
      avg_perturbed_real:扰动后真实数据得分均值
      decay_pct_real:    真实数据得分衰减率 (%)
    """
    perturb_params = [
        "pm_grid_down", "pm_grid_up",
        "pm_reduce_pct", "pm_add_pct",
        "pm_agg_k",
    ]

    # ── 基准综合得分 ──
    base_score = _compute_combined_score(
        params, base_kwargs, df_base, benchmark_return,
        scenario_dfs, scenario_benchmarks, sim_start_date)

    perturbed_scores = []
    for pname in perturb_params:
        base_val = params.get(pname)
        if base_val is None:
            continue

        # 向上扰动
        up_val = base_val * (1 + perturb_pct)
        if pname == "pm_reduce_pct":
            up_val = min(up_val, 0.95)
        elif pname == "pm_agg_k":
            up_val = min(up_val, 1.0)
        p_up = copy.deepcopy(params)
        p_up[pname] = round(up_val, 3)
        if _constraint_ok(p_up):
            try:
                up_score = _compute_combined_score(
                    p_up, base_kwargs, df_base, benchmark_return,
                    scenario_dfs, scenario_benchmarks, sim_start_date)
                perturbed_scores.append(up_score)
            except Exception:
                pass

        # 向下扰动
        down_val = base_val * (1 - perturb_pct)
        if pname == "pm_add_pct":
            down_val = max(down_val, 0.005)
        elif pname == "pm_agg_k":
            down_val = max(down_val, 0.2)
        p_down = copy.deepcopy(params)
        p_down[pname] = round(down_val, 3)
        if _constraint_ok(p_down):
            try:
                down_score = _compute_combined_score(
                    p_down, base_kwargs, df_base, benchmark_return,
                    scenario_dfs, scenario_benchmarks, sim_start_date)
                perturbed_scores.append(down_score)
            except Exception:
                pass

    if not perturbed_scores:
        return {"base_score": base_score, "avg_perturbed": base_score,
                "decay_pct": 0.0, "base_score_real": base_score,
                "avg_perturbed_real": base_score, "decay_pct_real": 0.0}

    avg_perturbed = float(np.mean(perturbed_scores))

    def _calc_decay(base: float, avg: float) -> float:
        if base > 0:
            return (base - avg) / base * 100
        elif abs(base) > 0.01:
            return (base - avg) / abs(base) * 100
        else:
            return 0.0

    decay_pct = _calc_decay(base_score, avg_perturbed)

    # ── 同时计算仅真实数据的衰减（用于对比）──
    if scenario_dfs:
        base_real = _compute_combined_score(
            params, base_kwargs, df_base, benchmark_return,
            None, None, None)
        perturbed_real = []
        for pname in perturb_params:
            base_val = params.get(pname)
            if base_val is None:
                continue
            for direction in [1, -1]:
                p_test = copy.deepcopy(params)
                new_val = base_val * (1 + direction * perturb_pct)
                if pname == "pm_reduce_pct":
                    new_val = min(new_val, 0.95)
                elif pname == "pm_add_pct":
                    new_val = max(new_val, 0.005)
                elif pname == "pm_agg_k":
                    new_val = max(min(new_val, 1.0), 0.2)
                p_test[pname] = round(new_val, 3)
                if _constraint_ok(p_test):
                    try:
                        s = _compute_combined_score(
                            p_test, base_kwargs, df_base, benchmark_return,
                            None, None, None)
                        perturbed_real.append(s)
                    except Exception:
                        pass
        if perturbed_real:
            avg_real = float(np.mean(perturbed_real))
            decay_real = _calc_decay(base_real, avg_real)
        else:
            base_real, avg_real, decay_real = base_score, base_score, 0.0
    else:
        base_real = base_score
        avg_real = avg_perturbed
        decay_real = decay_pct

    return {"base_score": base_score, "avg_perturbed": avg_perturbed,
            "decay_pct": decay_pct,
            "base_score_real": base_real, "avg_perturbed_real": avg_real,
            "decay_pct_real": decay_real}


def _constraint_ok(params: dict) -> bool:
    """检查参数是否满足逻辑约束"""
    gd = params.get("pm_grid_down", 0)
    gu = params.get("pm_grid_up", 0)
    rp = params.get("pm_reduce_pct", 0)
    ap = params.get("pm_add_pct", 0)
    return gd > gu and rp > ap


# ═══════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════

def grid_search():
    # ==================== 固定参数 ====================
    # base_kwargs = dict(
    #     ts_code="06181.HK",
    #     stock_name="老铺",
    #     market="HK",
    #     mode="position_management_1",
    #     per_shares=100,
    #     pm_start_date="20251001",
    #     start_date="20240628",
    #     user_dmi=True,
    #     pm_adaptive=True,
    #     pm_adaptive_grid=True,
    #     pm_dynamic_position=False,
    #     ema_fast=0,
    #     ema_slow=0,
    #     pm_initial_shares=3,
    #     initial_capital=40000,
    #     data_source="akshare",
    # )
    # base_kwargs = dict(
    #     ts_code="03690.HK",
    #     stock_name="美团",
    #     market="HK",
    #     mode="position_management_1",
    #     per_shares=100,
    #     pm_start_date="20220321",
    #     start_date="20180920",
    #     end_date="20220922",
    #     user_dmi=True,
    #     pm_adaptive=True,
    #     pm_adaptive_grid=True,
    #     pm_dynamic_position=False,
    #     ema_fast=0,
    #     ema_slow=0,
    #     pm_initial_shares=2,
    #     initial_capital=40000,
    #     data_source="akshare",
    # )
    # base_kwargs = dict(
    #     ts_code="02015.HK",
    #     stock_name="理想",
    #     market="HK",
    #     mode="position_management_1",
    #     per_shares=100,
    #     pm_start_date="20251001",
    #     start_date="20220813",
    #     user_dmi=True,
    #     pm_adaptive=True,
    #     pm_adaptive_grid=True,
    #     pm_dynamic_position=True,
    #     ema_fast=0,
    #     ema_slow=0,
    #     pm_initial_shares=25,
    #     initial_capital=40000,
    #     data_source="akshare",
    # )
    base_kwargs = dict(
            ts_code="09992.HK",
            stock_name="泡泡玛特",
            market="HK",
            mode="position_management_1",
            per_shares=200,
            pm_start_date="20251001",
            start_date="20201211",
            user_dmi=True,
            pm_adaptive=True,
            pm_adaptive_grid=True,
            pm_dynamic_position=False,
            ema_fast=0,
            ema_slow=0,
            pm_initial_shares=2,
            initial_capital=40000,
            data_source="akshare",
        )
   
    # ==================== 搜索空间 ====================
    # 趋势参数 (4维)
    pm_grid_down_vals       = [0.03, 0.05, 0.07, 0.10, 0.12]         # 5
    pm_grid_up_vals         = [0.03, 0.04, 0.05, 0.07, 0.10]         # 5
    pm_reduce_pct_vals      = [0.25, 0.35, 0.50, 0.75]               # 4
    pm_add_pct_vals         = [0.25, 0.30, 0.35, 0.40, 0.50]         # 5 (floor后1手≈3.2万,权益≈10万,需≥31%买1手)
    pm_adx_threshold_vals   = [25]                                  # 1 (固定，空头日短路无效)
    # 激进倍数 k: agg = f(趋势, k) 在引擎中自动派生
    k_vals                  = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]   # 7
    pm_take_profit_vals     = [1.30]                                  # 1 (固定)
    # 全量: 5×5×4×5×1×7×1 = 3,500，约束过滤后约 1,200

    search_lists = [
        pm_grid_down_vals, pm_grid_up_vals, pm_reduce_pct_vals, pm_add_pct_vals,
        pm_adx_threshold_vals, k_vals, pm_take_profit_vals,
    ]

    total = 1
    for lst in search_lists:
        total *= len(lst)
    print(f"🔍 搜索空间: {total:,} 种组合")
    print(f"  趋势参数: gd={pm_grid_down_vals}")
    print(f"           gu={pm_grid_up_vals}")
    print(f"           rp={pm_reduce_pct_vals}")
    print(f"           ap={pm_add_pct_vals}")
    print(f"           adx={pm_adx_threshold_vals}")
    print(f"  激进倍数: k={k_vals}  →  agg 由引擎派生 (gda=gd×k, gua=gu×k, rpa=rp+(1-k)×0.5)")
    print(f"  止盈倍数: {pm_take_profit_vals}")

    # ==================== 预加载数据 ====================
    print("\n📥 获取数据 & 计算指标 ...")
    config0 = BacktestConfig(**base_kwargs)
    df_base = fetch_data(
        ts_code=config0.ts_code, stock_name=config0.stock_name, market=config0.market,
        start_date=config0.start_date, end_date=config0.end_date,
        exchange=config0.exchange, data_source=config0.data_source,
    )
    df_base = compute_indicators(
        df_base, entry_period=config0.entry_period, exit_period=config0.exit_period,
        atr_period=config0.atr_period, ema_fast=config0.ema_fast, ema_slow=config0.ema_slow,
        channel_type=config0.channel_type, bb_period=config0.bb_period, bb_k=config0.bb_k,
        am_ma_period=config0.am_ma_period, mode=config0.mode,
    )
    print(f"   数据行数: {len(df_base)}")

    df_pm = df_base[df_base["Date"] >= config0.pm_start_date]
    buy_hold_return = ((df_pm["Close"].iloc[-1] - df_pm["Close"].iloc[0])
                       / df_pm["Close"].iloc[0] * 100) if len(df_pm) >= 2 else 0.0
    print(f"   买入持有收益率 (自{config0.pm_start_date}): {buy_hold_return:+.2f}%\n")

    # ═══════════════════════════════════════════════════
    # 第1轮: 粗筛 — 逻辑约束 + 基本质量过滤
    # ═══════════════════════════════════════════════════
    print("=" * 70)
    print("🔵 第1轮: 粗筛（逻辑约束 + 基本质量过滤）")
    print("=" * 70)
    print("  约束: gd>gu  rp>ap  (agg 由引擎根据 k 自动派生)")
    print("  过滤: 交易次数>0  最大回撤>-50%\n")

    all_results = []
    count = 0
    skipped_constraint = 0
    skipped_quality = 0
    t0 = time.time()

    for gd, gu, rp, ap, adx, k, tp in itertools.product(*search_lists):
        count += 1

        # ── 逻辑约束 ──
        if gd <= gu:
            skipped_constraint += 1
            continue
        if rp <= ap:
            skipped_constraint += 1
            continue

        params = dict(
            pm_grid_down=gd, pm_grid_up=gu, pm_reduce_pct=rp, pm_add_pct=ap,
            pm_adx_threshold=adx, pm_agg_k=k, pm_take_profit_ratio=tp,
        )
        config = BacktestConfig(**{**base_kwargs, **params})

        try:
            trades, equity, eq_dates = run_backtest(df_base.copy(), config)
            stats = calc_stats(trades, config.initial_capital, equity, eq_dates)

            n_trades = stats["总交易"]
            max_dd = stats["最大回撤 (%)"]

            # ── 基本质量过滤 ──
            if n_trades == 0:
                skipped_quality += 1
                continue
            if max_dd < -50:
                skipped_quality += 1
                continue

            total_return = stats["总收益率 (%)"]
            excess = total_return - buy_hold_return
            r3 = stats["R立方"]
            rs = stats["稳健夏普比率"]
            gda, gua, rpa = derive_agg(gd, gu, rp, k)

            all_results.append({
                **params,
                "gda": gda, "gua": gua, "rpa": rpa,
                "return": total_return,
                "excess": excess,
                "trades": n_trades,
                "max_dd": max_dd,
                "sharpe": stats["夏普比率"],
                "robust_sharpe": rs,
                "cagr": stats["CAGR (%)"],
                "final_capital": stats["最终资金 (¥)"],
                "r_cubed": r3,
                "score": 0.0,  # 占位，第2轮百分位归一化后填入
            })

        except Exception as e:
            if count % 500 == 0:
                print(f"   [{count:4d}/{total}] ⚠️ {e}")

        elapsed = time.time() - t0
        if count % 500 == 0:
            eta = (elapsed / count) * (total - count) if count > 0 else 0
            print(f"   [{count:4d}/{total}] {elapsed:.0f}s | ETA {eta/60:.0f}min | "
                  f"通过: {len(all_results)} | 约束过滤: {skipped_constraint} | 质量过滤: {skipped_quality}")

    elapsed = time.time() - t0
    print(f"\n✅ 第1轮完成: 通过 {len(all_results)} 组 "
          f"(约束过滤 {skipped_constraint}, 质量过滤 {skipped_quality}), "
          f"耗时 {elapsed/60:.1f} 分钟\n")

    # ═══════════════════════════════════════════════════
    # 第2轮: 排名 — 百分位归一化评分（不过滤，排名展示）
    # ═══════════════════════════════════════════════════
    print("=" * 70)
    print("🔵 第2a轮: 百分位归一化评分（超额50% + 稳健夏普40% + R³10%）")
    print("=" * 70)

    all_results = compute_percentile_scores(all_results)
    all_results.sort(key=lambda x: x["score"], reverse=True)

    print(f"\n  {'#':<3} {'得分':>7} {'超额%':>6} {'稳夏%':>5} "
          f"{'超额':>7} {'稳夏':>6} {'收益%':>7} {'笔':>3}  {'k':>5} 参数(gd/gu/rp/ap)")
    print(f"  {'─' * 100}")
    for i, r in enumerate(all_results[:20], 1):
        p = f"{r['pm_grid_down']:.2f}/{r['pm_grid_up']:.2f}/{r['pm_reduce_pct']:.2f}/" \
            f"{r['pm_add_pct']:.2f}"
        print(f"  {i:<3} {r['score']:>7.2f} {r.get('pct_excess',0):>6.0f} {r.get('pct_robust_sharpe',0):>5.0f} "
              f"{r['excess']:>7.2f} {r['robust_sharpe']:>6.2f} {r['return']:>7.2f} "
              f"{r['trades']:>3}  {r['pm_agg_k']:>4.2f}  {p}")
    if len(all_results) > 20:
        print(f"  ... (共 {len(all_results)} 组, 仅显示前20)")

    print(f"\n  得分范围: {all_results[-1]['score']:.2f} ~ {all_results[0]['score']:.2f}")
    print(f"  k 覆盖:   {min(r['pm_agg_k'] for r in all_results):.2f} ~ {max(r['pm_agg_k'] for r in all_results):.2f}")

    # ═══════════════════════════════════════════════════
    # 第2b轮: 滚动窗口交叉验证 — 取 TOP 200，在多子区间上验证，过滤时段过拟合
    # ═══════════════════════════════════════════════════
    CV_TOP_N = min(200, len(all_results))
    print(f"\n{'=' * 70}")
    print(f"🔵 第2b轮: 滚动窗口交叉验证（TOP {CV_TOP_N} → TOP 50）")
    print(f"{'=' * 70}")
    print(f"  窗口: 252天 × 步长63天, 取 P10 得分 → 惩罚特定时段过拟合\n")

    t_cv = time.time()
    top_candidates = all_results[:CV_TOP_N]
    for i, r in enumerate(top_candidates):
        params = {k: r[k] for k in r if k.startswith("pm_")}
        cv_score = rolling_window_cv(params, base_kwargs, df_base)
        r["cv_score"] = round(cv_score, 2)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t_cv
            eta = (elapsed / (i + 1)) * (len(top_candidates) - i - 1) if i > 0 else 0
            print(f"   已评估: {i+1}/{len(top_candidates)} ({elapsed:.0f}s | ETA {eta/60:.0f}min)")

    # 按 CV 得分降序，取 TOP 50 进入后续轮次
    top_candidates.sort(key=lambda x: x.get("cv_score", 0), reverse=True)
    all_results = top_candidates[:50]

    print(f"\n  ✅ 第2轮完成: 滚动CV筛选后保留 {len(all_results)} 组")
    print(f"  CV得分范围: {all_results[-1]['cv_score']:.2f} ~ {all_results[0]['cv_score']:.2f}")
    print(f"  耗时: {(time.time()-t_cv)/60:.1f} 分钟\n")

    # ═══════════════════════════════════════════════════
    # 第3轮: 验证 — 8种模拟行情压力测试
    # ═══════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("🔵 第3轮: 8种模拟行情压力测试（概率加权）")
    print(f"{'=' * 70}")
    print(f"  对全部 {len(all_results)} 组参数在8种场景上逐一回测...")

    from simulate_market_scenarios import get_last_price as sim_get_last_price
    try:
        last_price, last_date = sim_get_last_price()
    except Exception:
        from core.cache import load as cache_load
        cached = cache_load(config0.ts_code, config0.market)
        last_price = float(cached["Close"].iloc[-1])
        last_date = cached["Date"].iloc[-1]
    print(f"  最新收盘价: {last_price:.2f} HKD ({last_date})\n")

    print("  生成8种模拟行情数据...")
    scenario_dfs, scenario_benchmarks = prepare_scenario_dfs(last_price, last_date, config0, df_base)
    print(f"  场景数据准备完成\n")

    validated = []
    n_total = len(all_results)
    t1 = time.time()

    # ── 第一遍：收集所有场景原始指标（与真实数据评分口径统一）──
    # 存储: (idx, label, excess, robust_sharpe, return, r_cubed)
    raw_scenario_metrics = []
    for idx, r in enumerate(all_results):
        params = {k: r[k] for k in r if k.startswith("pm_")}
        for label in SCENARIO_LABEL_MAP:
            scenario_kwargs = {**base_kwargs, "pm_start_date": (last_date + timedelta(days=1)).strftime("%Y%m%d")}
            config_s = BacktestConfig(**{**scenario_kwargs, **params})
            try:
                trades_s, equity_s, eq_dates_s = run_backtest(
                    scenario_dfs[label].copy(), config_s)
                stats_s = calc_stats(trades_s, config_s.initial_capital,
                                     equity_s, eq_dates_s,
                                     benchmark_return=scenario_benchmarks[label])
                raw_scenario_metrics.append({
                    "idx": idx,
                    "label": label,
                    "excess": stats_s["超额收益 (%)"] or 0.0,
                    "robust_sharpe": stats_s["稳健夏普比率"] or 0.0,
                    "r_cubed": min(stats_s["R立方"] or 0.0, 10.0),
                    "total_return": stats_s["总收益率 (%)"],
                })
            except Exception:
                raw_scenario_metrics.append({
                    "idx": idx, "label": label,
                    "excess": float("-inf"), "robust_sharpe": float("-inf"),
                    "r_cubed": float("-inf"), "total_return": float("-inf"),
                })

        if (idx + 1) % 200 == 0:
            elapsed = time.time() - t1
            eta = (elapsed / (idx + 1)) * (n_total - idx - 1) if idx >= 0 else 0
            print(f"   已回测: {idx + 1}/{n_total} ({elapsed:.0f}s | ETA {eta/60:.0f}min)")

    # ── 第二遍：场景内每项指标百分位归一化 + 加权 + 概率汇总 ──
    # 权重（与真实数据评分一致）
    weights = {"excess": 0.50, "robust_sharpe": 0.40, "r_cubed": 0.10}
    df_raw = pd.DataFrame(raw_scenario_metrics)
    for metric in weights:
        df_raw[f"pct_{metric}"] = np.nan
    for label in SCENARIO_LABEL_MAP:
        for metric in weights:
            mask = (df_raw["label"] == label) & np.isfinite(df_raw[metric])
            if mask.any():
                df_raw.loc[mask, f"pct_{metric}"] = (
                    df_raw.loc[mask, metric].rank(pct=True) * 100)

    # 汇总：每个参数组合在每个场景的百分位加权分
    scenario_scores_by_idx = {}
    for _, row in df_raw.iterrows():
        idx = int(row["idx"])
        label = row["label"]
        if idx not in scenario_scores_by_idx:
            scenario_scores_by_idx[idx] = {}
        # 四维加权
        sw = sum(row.get(f"pct_{m}", 0) * w for m, w in weights.items()
                 if np.isfinite(row.get(f"pct_{m}", float("nan"))))
        scenario_scores_by_idx[idx][label] = round(sw, 2)

    for rank, r in enumerate(all_results):
        scores_by_label = scenario_scores_by_idx.get(rank, {})

        # 概率加权汇总
        valid_scores = []
        valid_probs = []
        for label in SCENARIO_LABEL_MAP:
            s = scores_by_label.get(label, float("-inf"))
            if np.isfinite(s):
                valid_scores.append(s)
                valid_probs.append(SCENARIO_PROBS[label])
        if valid_scores:
            total_prob = sum(valid_probs)
            weighted_score = sum(s * p / total_prob for s, p in zip(valid_scores, valid_probs))
        else:
            weighted_score = 0.0

        combined_score = r["score"] * 0.3 + weighted_score * 0.7

        # 熊市场景收益率
        slow_grind_ret = 0.0
        volatile_down_ret = 0.0
        for _, row in df_raw[df_raw["idx"] == rank].iterrows():
            if row["label"] == "缓慢阴跌":
                slow_grind_ret = row["total_return"]
            elif row["label"] == "震荡向下":
                volatile_down_ret = row["total_return"]
        worst_bear_ret = min(slow_grind_ret, volatile_down_ret)
        bear_crash = np.isfinite(worst_bear_ret) and worst_bear_ret < -25

        entry = {
            **r,
            "weighted_score": round(weighted_score, 2),
            "combined_score": round(combined_score, 2),
            "slow_grind_ret": slow_grind_ret,
            "volatile_down_ret": volatile_down_ret,
            "bear_crash": bear_crash,
        }
        for label in SCENARIO_LABEL_MAP:
            entry[f"sim_{label}"] = scores_by_label.get(label, float("nan"))
        validated.append(entry)

    print(f"  场景验证耗时: {(time.time()-t1)/60:.1f} 分钟")

    robust_candidates = [v for v in validated if not v["bear_crash"]]
    if len(robust_candidates) < 5:
        print(f"  ⚠️ 熊市过滤后仅剩 {len(robust_candidates)} 组，放宽条件")
        robust_candidates = validated

    robust_candidates.sort(key=lambda x: x["combined_score"], reverse=True)
    top10 = robust_candidates[:10]

    print(f"\n  第3轮 TOP 10（真实30% + 模拟70%）:")
    print(f"  {'#':<3} {'综合':>7} {'真实':>7} {'模拟':>7} "
          f"{'缓跌%':>7} {'震下%':>7}  {'k':>5}  参数")
    print(f"  {'─' * 90}")
    for i, v in enumerate(top10, 1):
        p = f"gd={v['pm_grid_down']:.2f} gu={v['pm_grid_up']:.2f} " \
            f"rp={v['pm_reduce_pct']:.2f} ap={v['pm_add_pct']:.2f} " \
            f"adx={v['pm_adx_threshold']:.0f}"
        print(f"  {i:<3} {v['combined_score']:>7.2f} {v['score']:>7.2f} "
              f"{v['weighted_score']:>7.2f} {v['slow_grind_ret']:>7.2f} "
              f"{v['volatile_down_ret']:>7.2f}  {v['pm_agg_k']:>4.2f}  {p}")

    # ═══════════════════════════════════════════════════
    # 第4轮: 鲁棒 — 参数敏感性检验（真实 + 模拟）
    # ═══════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("🔵 第4轮: 参数敏感性检验（±15%，真实+模拟综合衰减）")
    print(f"{'=' * 70}")
    print(f"  同时扰动真实数据和8种模拟行情，计算综合得分衰减率")
    print(f"  综合得分 = 真实数据得分×0.3 + 模拟概率加权得分×0.7")

    sim_start = (last_date + timedelta(days=1)).strftime("%Y%m%d")

    # 从第3轮候选顶部开始，逐批检验直到找出 ≥3 组稳定组合
    final_candidates = []
    batch_size = 10
    t2 = time.time()

    for batch_start in range(0, len(robust_candidates), batch_size):
        batch = robust_candidates[batch_start:batch_start + batch_size]
        stable_in_batch = 0
        print(f"\n  ── 批次 {batch_start//batch_size + 1} (第{batch_start+1}-{batch_start+len(batch)}名) ──")

        for i, v in enumerate(batch, 1):
            rank = batch_start + i
            params = {k: v[k] for k in v if k.startswith("pm_")}
            sens = sensitivity_test(params, base_kwargs, df_base,
                                    benchmark_return=buy_hold_return,
                                    scenario_dfs=scenario_dfs,
                                    scenario_benchmarks=scenario_benchmarks,
                                    sim_start_date=sim_start)
            decay = sens["decay_pct"]
            decay_real = sens["decay_pct_real"]
            stable = decay < 10

            status = "✅ 稳定" if stable else "❌ 敏感"
            print(f"  {rank:>3}. 综合基准={sens['base_score']:.2f}  扰动均值={sens['avg_perturbed']:.2f}  "
                  f"衰减={decay:+.1f}%  "
                  f"(真实:{decay_real:+.1f}%)  {status}")
            print(f"       gd={v['pm_grid_down']:.2f} gu={v['pm_grid_up']:.2f} "
                  f"rp={v['pm_reduce_pct']:.2f} ap={v['pm_add_pct']:.2f} "
                  f"k={v['pm_agg_k']:.2f}")

            final_candidates.append({**v, "decay_pct": decay, "decay_pct_real": decay_real,
                                     "stable": stable})
            if stable:
                stable_in_batch += 1

        # 累计稳定组合数
        total_stable = sum(1 for c in final_candidates if c["stable"])
        print(f"  累计稳定: {total_stable} 组")

        if total_stable >= 3:
            break
        if batch_start + batch_size >= len(robust_candidates):
            print(f"  已遍历全部 {len(robust_candidates)} 组候选")
            break

    print(f"\n  敏感性检验耗时: {(time.time()-t2)/60:.1f} 分钟")

    # 取稳定组合，加入稳定性惩罚后重新排名
    stable_final = [c for c in final_candidates if c["stable"]]
    if len(stable_final) < 2:
        print(f"  ⚠️ 稳定组合不足2组（共 {len(stable_final)} 组），取衰减最小的3组")
        final_candidates.sort(key=lambda x: x["decay_pct"])
        stable_final = final_candidates[:3]
    else:
        # Plan A: 稳定性惩罚内嵌到评分 — 衰减超5%开始惩罚，指数递增
        for c in stable_final:
            penalty = max(0, c["decay_pct"] - 5) ** 1.5
            c["stability_penalty"] = round(penalty, 2)
            c["robust_score"] = round(c["combined_score"] - penalty, 2)
        stable_final.sort(key=lambda x: x["robust_score"], reverse=True)
        stable_final = stable_final[:3]

    # ═══════════════════════════════════════════════════
    # 最终结果
    # ═══════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print("🏆 最终候选参数（通过全部4轮筛选）")
    print(f"{'=' * 60}")

    for i, c in enumerate(stable_final, 1):
        gd = c['pm_grid_down']
        gu = c['pm_grid_up']
        rp = c['pm_reduce_pct']
        ap = c['pm_add_pct']
        adx = int(c['pm_adx_threshold'])
        k = c['pm_agg_k']
        gda, gua, rpa = c['gda'], c['gua'], c['rpa']
        tp = c.get('pm_take_profit_ratio', 1.30)

        print(f"\n  ── 候选 #{i} ──")
        print(f"   pm_grid_down       = {gd:.2f}")
        print(f"   pm_grid_up         = {gu:.2f}")
        print(f"   pm_reduce_pct      = {rp:.2f}")
        print(f"   pm_add_pct         = {ap:.2f}")
        print(f"   pm_adx_threshold   = {adx}")
        print(f"   pm_agg_k           = {k:.2f}")
        print(f"   ── 派生的 agg 参数 ──")
        print(f"   pm_grid_down_agg   = {gda:.2f}  (gd × {k:.2f})")
        print(f"   pm_grid_up_agg     = {gua:.2f}  (gu × {k:.2f})")
        print(f"   pm_reduce_pct_agg  = {rpa:.2f}  (rp + (1-{k:.2f})×0.5)")
        print(f"   pm_take_profit_ratio = {tp:.2f}")
        print(f"   ──────────────────")
        print(f"   真实回测: 收益={c['return']:+.2f}%  超额={c['excess']:+.2f}%  "
              f"回撤={c['max_dd']:.2f}%  R³={c['r_cubed']:.2f}  百分位得分={c['score']:.2f}")
        print(f"   滚动CV:   得分={c.get('cv_score', 'N/A')}")
        print(f"   模拟加权: 得分={c['weighted_score']:.2f}  综合={c['combined_score']:.2f}")
        print(f"   敏感性:   综合衰减={c['decay_pct']:+.1f}%  "
              f"真实衰减={c.get('decay_pct_real', c['decay_pct']):+.1f}%  "
              f"{'✅ 稳定' if c['stable'] else '⚠️ 略敏感'}")
        if c.get('stability_penalty', 0) > 0:
            print(f"   稳定性惩罚: -{c['stability_penalty']:.2f}  →  鲁棒得分: {c['robust_score']:.2f}")

    # ═══════════════════════════════════════════════════
    # 保存 Excel
    # ═══════════════════════════════════════════════════
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "result")
    os.makedirs(out_dir, exist_ok=True)

    pd.DataFrame(all_results).to_excel(
        os.path.join(out_dir, "pm1_grid_search_all.xlsx"), index=False)

    pd.DataFrame(validated).to_excel(
        os.path.join(out_dir, "pm1_grid_search_validated.xlsx"), index=False)

    pd.DataFrame(stable_final).to_excel(
        os.path.join(out_dir, "pm1_grid_search_final.xlsx"), index=False)

    print(f"\n📁 结果已保存:")
    print(f"   全量:   result/pm1_grid_search_all.xlsx ({len(all_results)} 组)")
    print(f"   验证:   result/pm1_grid_search_validated.xlsx ({len(validated)} 组)")
    print(f"   最终:   result/pm1_grid_search_final.xlsx ({len(stable_final)} 组)")
    print(f"\n✅ 四轮优化完成！")


if __name__ == "__main__":
    grid_search()
