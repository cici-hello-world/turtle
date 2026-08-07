#!/usr/bin/env python3
"""场景压力测试报告 — 对最优参数在8种模拟行情上逐一回测，输出风险评估。

不参与参数排名。用于在参数选定后回答：
  • 这个参数在什么市况下会失效？
  • 当前市场判断下，策略大概率表现如何？
  • 什么情况下应该考虑降仓或暂停策略？

用法：
  python strategy/position_management_1/scenario_stress_report.py --ts-code 06181.HK
  python strategy/position_management_1/scenario_stress_report.py --ts-code 06181.HK --top-n 3
"""

import argparse
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
from core.engine import run_backtest
from core.statistics import calc_stats

from strategy.position_management_1 import common_config
from strategy.position_management_1.grid_search_pm import (
    prepare_scenario_dfs,
    SCENARIO_LABEL_MAP,
    get_scenario_probs,
)
from strategy.position_management_1.simulate_market_scenarios import (
    get_last_price as sim_get_last_price,
)

# ── scenario_analysis 概率（优先使用实时分析结果）──
try:
    from strategy.position_management_1.scenario_analysis import (
        load_data, match_scenarios, estimate_probabilities,
    )
    _HAS_SCENARIO_ANALYSIS = True
except ImportError:
    _HAS_SCENARIO_ANALYSIS = False


def get_live_probs(ts_code: str) -> dict | None:
    """尝试获取实时场景概率，失败返回 None。"""
    if not _HAS_SCENARIO_ANALYSIS:
        return None
    try:
        df = load_data(ts_code)
        result = match_scenarios(df, "2025-10-01")
        if result:
            estimates = estimate_probabilities(
                result["scores"], ts_code, features=result["features"])
            return estimates["probs"]
    except Exception:
        pass
    return None


def print_report(ts_code: str, stock_name: str, param_sets: list[dict],
                 scenario_results: list[dict], probs: dict,
                 latest_price: float, save_path: str | None = None):
    """打印压力测试报告。"""

    lines = []
    def emit(s=""):
        lines.append(s)
        print(s)

    emit("=" * 85)
    emit(f"  🏆 {stock_name} ({ts_code})  最优参数 场景压力测试报告")
    emit(f"  最新价: ¥{latest_price:.2f}")
    emit("=" * 85)

    # ── 参数 ──
    for pi, pset in enumerate(param_sets):
        emit(f"\n  参数组 #{pi + 1}:")
        emit(f"    gd={pset.get('pm_grid_down', '?'):.3f}  "
             f"gu={pset.get('pm_grid_up', '?'):.3f}  "
             f"rp={pset.get('pm_reduce_pct', '?'):.3f}  "
             f"ap={pset.get('pm_add_pct', '?'):.3f}  "
             f"adx={pset.get('pm_adx_threshold', '?'):.0f}  "
             f"k={pset.get('pm_agg_k', '?'):.3f}")

    # ── 当前市场判断 ──
    emit(f"\n  {'─' * 75}")
    if probs:
        top3 = sorted(probs.items(), key=lambda x: -x[1])[:3]
        prob_str = "  ".join(f"{n} {p:.0%}" for n, p in top3)
        emit(f"  当前市场判断: {prob_str}")
    else:
        emit(f"  当前市场判断: (无法获取实时概率，使用硬编码概率)")

    # ── 场景结果表 ──
    emit(f"\n  {'场景':<10s} {'概率':>6s} {'价格变动':>8s} {'策略收益':>8s} "
         f"{'超额收益':>8s} {'最大回撤':>8s} {'交易':>5s} {'夏普':>6s}  {'判定'}")
    emit(f"  {'─' * 82}")

    for sr in scenario_results:
        label = sr["label"]
        prob = probs.get(label, 0)
        s = sr["stats"]
        price_chg = sr.get("benchmark", 0)

        # 判定
        excess = s.get("超额收益 (%)", 0) or 0
        strat_ret = s.get("总收益率 (%)", 0) or 0
        if strat_ret < -25:
            verdict = "❌ 极端"
        elif strat_ret < -10:
            verdict = "❌ 脆弱"
        elif strat_ret < 0:
            verdict = "⚠️ 打平"
        elif strat_ret < 10:
            verdict = "✅"
        else:
            verdict = "✅ 优秀"

        arrow = ("📉" if ("跌" in label or "倒V" in label)
                 else ("📈" if ("涨" in label or ("V" in label and "倒" not in label))
                       else "📊"))

        emit(f"  {arrow} {label:<8s} {prob:>5.0%}  {price_chg:>+7.1f}%  "
             f"{strat_ret:>+7.1f}%  {excess:>+7.1f}%  "
             f"{s.get('最大回撤 (%)', 0):>+7.1f}%  "
             f"{s.get('总交易', 0):>5d}  {s.get('夏普比率', 0):>+5.2f}  "
             f"{verdict}")

    # ── 风险评估 ──
    emit(f"\n  {'─' * 75}")
    emit(f"  📋 风险评估")
    emit(f"  {'─' * 75}")

    # 找到主导场景和脆弱场景
    sorted_by_prob = sorted(probs.items(), key=lambda x: -x[1]) if probs else []
    dominant = sorted_by_prob[0] if sorted_by_prob else None
    vulnerable = [(n, p) for n, p in sorted_by_prob
                  if any(sr["label"] == n and (sr["stats"].get("总收益率 (%)", 0) or 0) < -10
                         for sr in scenario_results)]
    extreme = [(n, p) for n, p in sorted_by_prob
               if any(sr["label"] == n and (sr["stats"].get("总收益率 (%)", 0) or 0) < -25
                      for sr in scenario_results)]

    if dominant:
        dom_name, dom_prob = dominant
        dom_sr = next((sr for sr in scenario_results if sr["label"] == dom_name), None)
        if dom_sr:
            dom_ret = dom_sr["stats"].get("总收益率 (%)", 0) or 0
            dom_excess = dom_sr["stats"].get("超额收益 (%)", 0) or 0
            if dom_ret >= 0:
                emit(f"\n  ✅ 主导场景（{dom_name} {dom_prob:.0%}）："
                     f"策略收益 {dom_ret:+.1f}%，超额 {dom_excess:+.1f}%。"
                     f"大概率盈利或打平。")
            elif dom_ret > -10:
                emit(f"\n  ⚠️ 主导场景（{dom_name} {dom_prob:.0%}）："
                     f"策略收益 {dom_ret:+.1f}%，小幅亏损。需密切关注。")
            else:
                emit(f"\n  ❌ 主导场景（{dom_name} {dom_prob:.0%}）："
                     f"策略收益 {dom_ret:+.1f}%，显著亏损。应考虑降仓。")

    if vulnerable:
        vuln_str = "、".join(f"{n}({p:.0%})" for n, p in vulnerable)
        emit(f"\n  ❌ 脆弱场景: {vuln_str}")
        emit(f"     网格策略在这些市况下结构性失效——反弹不足，加仓即套。")

    if extreme and extreme != vulnerable:
        extr_str = "、".join(f"{n}({p:.0%})" for n, p in extreme)
        emit(f"\n  💀 极端场景: {extr_str}")
        emit(f"     理论最大亏损。真实 DMI 风控会提前止损，实际亏损应小于模拟。")

    # ── 一句话建议 ──
    emit(f"\n  {'─' * 75}")
    if dominant and dom_name in dict(vulnerable):
        emit(f"  💡 建议: 主导场景本身偏空，策略在该市况下脆弱。"
             f"考虑减仓或暂停网格，等待市况切换。")
    elif vulnerable:
        vuln_probs = sum(p for _, p in vulnerable)
        if vuln_probs > 0.3:
            emit(f"  💡 建议: 脆弱场景合计概率 {vuln_probs:.0%}，偏高。"
                 f"建议设定降仓触发条件（如连续阴跌 N 日）。")
        else:
            emit(f"  💡 建议: 脆弱场景合计概率仅 {vuln_probs:.0%}，整体风险可控。"
                 f"若市场实际转入阴跌，应触发降仓。")
    else:
        emit(f"  💡 建议: 各场景下策略表现稳健，按正常仓位运行。")

    emit(f"\n{'=' * 85}")

    # ── 保存 ──
    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"\n📁 报告已保存: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="场景压力测试报告 — 对最优参数跑8种模拟行情")
    parser.add_argument("--ts-code", type=str, default="09992.HK",
                        help="股票代码 (默认: 09992.HK)")
    parser.add_argument("--top-n", type=int, default=1,
                        help="报告前 N 组最优参数 (默认: 1)")
    parser.add_argument("--save", action="store_true",
                        help="保存报告到 result/ 目录")
    args = parser.parse_args()

    ts_code = args.ts_code

    # ── 激活配置 ──
    try:
        common_config.init_config(ts_code)
    except KeyError as e:
        print(f"❌ init_config 失败 ({e})")
        sys.exit(1)

    stock_name = common_config.STOCKS[ts_code]["stock_name"]
    config0 = BacktestConfig(**common_config.BASE_KWARGS)

    # ── 获取最优参数 ──
    optimal = common_config.OPTIMAL_PARAMS_ACTIVE
    if optimal is None:
        print(f"❌ {ts_code} 没有最优参数，请先运行 turtle_scoring")
        sys.exit(1)

    # 构建参数列表
    param_keys = ["pm_grid_down", "pm_grid_up", "pm_reduce_pct",
                  "pm_add_pct", "pm_adx_threshold", "pm_agg_k"]
    top_params = [{k: optimal.get(k) for k in param_keys}]
    # 如果 optimal 是列表（多组参数），取前 top_n
    if isinstance(optimal, list) and len(optimal) > 1:
        top_params = [{k: p.get(k) for k in param_keys} for p in optimal[:args.top_n]]
    top_params = top_params[:args.top_n]

    # ── 加载数据 ──
    print(f"\n🚀 加载 {stock_name} ({ts_code}) 数据 ...")
    df = fetch_data(
        ts_code=config0.ts_code, stock_name=config0.stock_name,
        market=config0.market, start_date=config0.start_date,
        end_date=config0.end_date, exchange=config0.exchange,
        data_source=config0.data_source,
    )
    df = compute_indicators(
        df,
        entry_period=config0.entry_period, exit_period=config0.exit_period,
        atr_period=config0.atr_period, ema_fast=config0.ema_fast,
        ema_slow=config0.ema_slow, channel_type=config0.channel_type,
        bb_period=config0.bb_period, bb_k=config0.bb_k,
        am_ma_period=config0.am_ma_period, mode=config0.mode,
    )

    # ── 获取最新价 + 日期 ──
    try:
        last_price, last_date = sim_get_last_price()
    except Exception:
        from core.cache import load as cache_load
        cached = cache_load(config0.ts_code, config0.market)
        last_price = float(cached["Close"].iloc[-1])
        last_date = cached["Date"].iloc[-1]

    # ── 生成场景数据 ──
    print(f"  生成8种模拟行情数据 ...")
    t0 = time.time()
    scenario_dfs, scenario_benchmarks = prepare_scenario_dfs(
        last_price, last_date, config0, df)
    print(f"  场景数据准备完成 ({(time.time()-t0):.0f}s)")

    # ── 获取场景概率（优先实时）──
    print(f"  获取场景概率 ...")
    probs = get_live_probs(ts_code)
    if probs is None:
        try:
            probs = get_scenario_probs(ts_code)
            print(f"  使用硬编码概率")
        except KeyError:
            probs = {label: 0.125 for label in SCENARIO_LABEL_MAP}
            print(f"  使用均匀分布")
    else:
        print(f"  使用实时 scenario_analysis 概率")

    # ── 对每组参数跑8场景回测 ──
    sim_start = (last_date + timedelta(days=1)).strftime("%Y%m%d")

    all_scenario_results = []
    for pi, params in enumerate(top_params):
        print(f"\n  跑参数组 #{pi+1} 的 8 场景回测 ...")
        scenario_results = []
        for label in SCENARIO_LABEL_MAP:
            scenario_kwargs = {**common_config.BASE_KWARGS,
                              "pm_start_date": sim_start}
            config_s = BacktestConfig(**{**scenario_kwargs, **params})
            try:
                trades_s, equity_s, eq_dates_s = run_backtest(
                    scenario_dfs[label].copy(), config_s)
                stats_s = calc_stats(trades_s, config_s.initial_capital,
                                     equity_s, eq_dates_s,
                                     benchmark_return=scenario_benchmarks.get(label, 0.0))
                scenario_results.append({
                    "label": label,
                    "stats": stats_s,
                    "benchmark": scenario_benchmarks.get(label, 0.0),
                })
            except Exception as e:
                print(f"    ⚠️ {label} 回测失败: {e}")
                scenario_results.append({
                    "label": label,
                    "stats": {"总收益率 (%)": 0, "超额收益 (%)": 0,
                              "最大回撤 (%)": 0, "总交易": 0, "夏普比率": 0},
                    "benchmark": scenario_benchmarks.get(label, 0.0),
                })

        all_scenario_results.append(scenario_results)

    # ── 输出报告 ──
    for pi, scenario_results in enumerate(all_scenario_results):
        save_path = None
        if args.save:
            out_dir = os.path.join(PROJECT_ROOT, "result")
            os.makedirs(out_dir, exist_ok=True)
            today = pd.Timestamp.today().strftime("%Y%m%d")
            suffix = f"_{pi+1}" if len(all_scenario_results) > 1 else ""
            save_path = os.path.join(
                out_dir, f"scenario_stress_{ts_code.replace('.', '_')}_{today}{suffix}.txt")

        print_report(
            ts_code=ts_code,
            stock_name=stock_name,
            param_sets=[top_params[pi]],
            scenario_results=scenario_results,
            probs=probs,
            latest_price=last_price,
            save_path=save_path,
        )


if __name__ == "__main__":
    main()
