#!/usr/bin/env python3
"""BB宽度参数调优 —— 独立于主 sweep 流水线，按需运行。

用法：
    python tune_bb_params.py --ts-code 09992.HK
    python tune_bb_params.py --ts-code 06181.HK --rounds 3

搜索空间：
    pm_bb_period          BB 计算周期
    pm_bb_k               BB 标准差倍数
    pm_bb_width_lookback  BB 宽度 Z-score 归一化回看窗口
    pm_bb_width_min       窄带时网格乘数下限
    pm_bb_width_max       宽带时网格乘数上限

策略：多轮收缩搜索 —— 每轮在前一轮最优值附近缩小范围。
"""

import argparse
import itertools
import time

from core.config import BacktestConfig
from core.data import fetch_data
from core.indicators import compute_indicators
from core.engine import run_backtest
from strategy.position_management_1 import common_config


# ── 默认搜索空间（第1轮） ──
DEFAULT_SEARCH_SPACE = {
    "pm_bb_period":          [5, 10, 20],
    "pm_bb_k":               [1.5, 2.0, 2.5],
    "pm_bb_width_lookback":  [21, 42, 63, 126, 252],
    "pm_bb_width_min":       [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
    "pm_bb_width_max":       [1.05, 1.10, 1.15, 1.20, 1.30, 1.40, 1.50],
}

# ── 每轮收缩比例（在前一轮最优值附近 ±range_shrink 范围内搜索） ──
RANGE_SHRINK = 0.5

# ── 约束 ──
def _ok(min_val, max_val):
    return min_val < max_val


def _build_combos(space: dict, prev_best: dict | None = None,
                  shrink: float = RANGE_SHRINK) -> list[dict]:
    """生成参数组合列表（笛卡尔积 + 约束过滤）。

    如果提供 prev_best，则在最优值附近收缩搜索范围。
    """
    param_names = list(space.keys())
    lists = []
    for p in param_names:
        if prev_best and p in prev_best:
            best_v = prev_best[p]
            full = space[p]
            # 在最优值附近缩半范围
            half_span = (max(full) - min(full)) * shrink / 2
            narrowed = sorted(set(
                v for v in full
                if best_v - half_span <= v <= best_v + half_span
            ))
            lists.append(narrowed if narrowed else full)
        else:
            lists.append(space[p])

    combos = []
    for values in itertools.product(*lists):
        params = dict(zip(param_names, values))
        if _ok(params["pm_bb_width_min"], params["pm_bb_width_max"]):
            combos.append(params)
    return combos


def evaluate(ts_code: str, stock_name: str, market: str,
             base_kwargs: dict, optimal_params: dict,
             bb_combos: list[dict]) -> list[dict]:
    """评估所有BB参数组合，返回按 final_eq 降序的结果列表。"""
    # 预加载数据（只需要一次）
    config0 = BacktestConfig(**{**base_kwargs, **optimal_params})
    df = fetch_data(
        ts_code=ts_code, stock_name=stock_name, market=market,
        start_date=config0.start_date, end_date=config0.end_date,
        exchange=config0.exchange, data_source=config0.data_source,
    )

    results = []
    total = len(bb_combos)
    t0 = time.time()
    for i, bb in enumerate(bb_combos):
        kw = {**base_kwargs, **optimal_params, "pm_bb_enabled": True, **bb}
        config = BacktestConfig(**kw)

        df_ind = compute_indicators(df.copy(),
            entry_period=config.entry_period, exit_period=config.exit_period,
            atr_period=config.atr_period, ema_fast=config.ema_fast,
            ema_slow=config.ema_slow, channel_type=config.channel_type,
            bb_period=config.bb_period, bb_k=config.bb_k,
            am_ma_period=config.am_ma_period, mode=config.mode,
            pm_bb_period=config.pm_bb_period, pm_bb_k=config.pm_bb_k,
            pm_bb_width_lookback=config.pm_bb_width_lookback,
        )

        trades, eq, _ = run_backtest(df_ind, config)
        if not eq:
            continue

        initial = (config.initial_capital
                   + config.pm_initial_shares
                   * df_ind["Close"].iloc[0]
                   * config.per_shares)
        final_eq = eq[-1]
        pnl_pct = (final_eq / initial - 1) * 100

        # 最大回撤
        peak = eq[0]
        max_dd = 0.0
        for v in eq:
            if v > peak:
                peak = v
            dd = (peak - v) / peak * 100
            if dd > max_dd:
                max_dd = dd

        results.append({
            "params": bb,
            "trades": len(trades),
            "final_eq": final_eq,
            "pnl_pct": pnl_pct,
            "max_dd": max_dd,
        })

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{total}] {elapsed:.0f}s ...")

    results.sort(key=lambda x: x["final_eq"], reverse=True)
    return results


def print_results(results: list[dict], top_n: int = 5):
    """打印前 top_n 结果。"""
    print(f"\n{'Rank':<5s} {'final_eq':>10s} {'P&L%':>7s} {'MaxDD':>7s} {'Trades':>6s}  Params")
    print("-" * 85)
    for rank, r in enumerate(results[:top_n], 1):
        params_str = ", ".join(
            f"{k}={v}" for k, v in r["params"].items()
        )
        print(f"{rank:<5d} {r['final_eq']:>10,.0f} {r['pnl_pct']:>6.1f}% "
              f"{r['max_dd']:>6.1f}% {r['trades']:>6d}  {params_str}")


def main():
    parser = argparse.ArgumentParser(description="BB宽度参数调优")
    parser.add_argument("--ts-code", type=str, required=True,
                        help="股票代码，如 09992.HK")
    parser.add_argument("--rounds", type=int, default=2,
                        help="收缩搜索轮数 (默认: 2)")
    parser.add_argument("--top-n", type=int, default=5,
                        help="每轮打印前 N 个结果 (默认: 5)")
    args = parser.parse_args()

    ts_code = args.ts_code
    common_config.init_config(ts_code)
    base_kwargs = dict(common_config.BASE_KWARGS)
    optimal_params = dict(common_config.OPTIMAL_PARAMS_ACTIVE)
    stock_name = base_kwargs.get("stock_name", ts_code)
    market = base_kwargs.get("market", "HK")

    # ── 基线（BB OFF） ──
    print(f"\n{'='*60}")
    print(f"BB 参数调优 — {stock_name} {ts_code}")
    print(f"{'='*60}")

    kw_off = {**base_kwargs, **optimal_params, "pm_bb_enabled": False}
    config_off = BacktestConfig(**kw_off)
    df = fetch_data(ts_code=ts_code, stock_name=stock_name, market=market,
                    start_date=config_off.start_date, end_date=config_off.end_date,
                    exchange=config_off.exchange, data_source=config_off.data_source)
    df = compute_indicators(df,
        entry_period=config_off.entry_period, exit_period=config_off.exit_period,
        atr_period=config_off.atr_period, ema_fast=config_off.ema_fast,
        ema_slow=config_off.ema_slow, channel_type=config_off.channel_type,
        bb_period=config_off.bb_period, bb_k=config_off.bb_k,
        am_ma_period=config_off.am_ma_period, mode=config_off.mode,
        pm_bb_period=config_off.pm_bb_period, pm_bb_k=config_off.pm_bb_k,
        pm_bb_width_lookback=config_off.pm_bb_width_lookback,
    )
    trades_off, eq_off, _ = run_backtest(df, config_off)
    initial_off = (config_off.initial_capital
                   + config_off.pm_initial_shares
                   * df["Close"].iloc[0] * config_off.per_shares)
    pnl_off = (eq_off[-1] / initial_off - 1) * 100 if eq_off else 0
    peak_off = eq_off[0] if eq_off else 0
    max_dd_off = 0.0
    for v in eq_off:
        if v > peak_off: peak_off = v
        dd = (peak_off - v) / peak_off * 100
        if dd > max_dd_off: max_dd_off = dd

    print(f"\n基线 (BB OFF): trades={len(trades_off)}, "
          f"final_eq={eq_off[-1]:,.0f}, P&L={pnl_off:.1f}%, "
          f"MaxDD={max_dd_off:.1f}%")

    # ── 多轮收缩搜索 ──
    space = dict(DEFAULT_SEARCH_SPACE)
    prev_best = None

    for round_num in range(1, args.rounds + 1):
        combos = _build_combos(space, prev_best)
        print(f"\n--- 第 {round_num} 轮: {len(combos)} 个组合 ---")

        results = evaluate(ts_code, stock_name, market,
                          base_kwargs, optimal_params, combos)
        if not results:
            print("  ⚠️ 无有效结果")
            break

        print_results(results, args.top_n)

        best = results[0]
        prev_best = best["params"]
        print(f"\n  ✅ 第{round_num}轮最优: final_eq={best['final_eq']:,.0f} "
              f"(vs OFF: {best['final_eq'] - eq_off[-1]:+,.0f})")

    # ── 最终建议 ──
    print(f"\n{'='*60}")
    print("建议添加到 STOCKS 配置:")
    best_params = results[0]["params"] if results else {}
    for k in ["pm_bb_period", "pm_bb_k", "pm_bb_width_lookback",
              "pm_bb_width_min", "pm_bb_width_max"]:
        if k in best_params:
            print(f"  {k}={best_params[k]},")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
