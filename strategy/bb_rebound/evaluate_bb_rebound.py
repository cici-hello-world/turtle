#!/usr/bin/env python3
"""布林带返弹策略参数评估

扫描 bb_period × bb_k × pm_add_pct × pm_reduce_pct 网格（穿越触发），
对照「PM1 最优参数」与「买入持有」，给出推荐加减仓参数。

用法：
    python strategy/bb_rebound/evaluate_bb_rebound.py --ts-code 02015.HK
    python strategy/bb_rebound/evaluate_bb_rebound.py --ts-code 01810.HK

初始仓位/现金/日期从 common_config.STOCKS 取（与 PM1 一致）。
数据来源：data_cache 缓存（end_date 固定 20260812，不联网）。
"""

import argparse
import itertools
import os
import sys
from concurrent.futures import ProcessPoolExecutor

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


def _base_kwargs(ts_code: str) -> dict:
    """从 common_config.STOCKS 取标的初始仓位/现金/日期，切到 bb_rebound 模式"""
    if ts_code not in common_config.STOCKS:
        raise KeyError(f"未配置 {ts_code}，请先在 common_config.STOCKS 中添加")
    base = dict(common_config.STOCKS[ts_code])
    base.update(
        mode="bb_rebound",
        end_date="20260812",              # 只用缓存数据，不联网
        pm_max_position_pct=0.9,          # 仓位上限：市值不超过权益 90%
        pm_min_cash_pct=0.10,             # 现金下限：保留权益 10% 现金
        pm_take_profit_ratio=100,         # 默认关闭止盈清仓
    )
    return base

# ── 扫描网格 ──
GRID = dict(
    bb_period=[10, 20, 30],
    bb_k=[1.5, 2.0, 2.5],
    pm_add_pct=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50],
    pm_reduce_pct=[0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.75],
)

# ── 多进程共享数据 ──
_G_DF = None
_G_KWARGS = None
_G_BENCH = None


def _init_worker(df, base_kwargs, bench):
    global _G_DF, _G_KWARGS, _G_BENCH
    _G_DF, _G_KWARGS, _G_BENCH = df, base_kwargs, bench


def _run_one(params: dict) -> dict | None:
    """单组参数回测，返回汇总指标 dict（与 turtle_optimize.run_one_backtest 同模式）"""
    cfg = BacktestConfig(**{**_G_KWARGS, **params})
    df_ind = compute_indicators(_G_DF.copy(),
        entry_period=cfg.entry_period, exit_period=cfg.exit_period,
        atr_period=cfg.atr_period, ema_fast=cfg.ema_fast, ema_slow=cfg.ema_slow,
        channel_type=cfg.channel_type, bb_period=cfg.bb_period, bb_k=cfg.bb_k,
        am_ma_period=cfg.am_ma_period, mode=cfg.mode,
        pm_bb_period=cfg.pm_bb_period, pm_bb_k=cfg.pm_bb_k,
        pm_bb_width_lookback=cfg.pm_bb_width_lookback)
    trades, equity, eq_dates = run_backtest(df_ind, cfg)
    if not equity or len(equity) < 2:
        return None
    stats = calc_stats(trades, cfg.initial_capital, equity, eq_dates,
                       benchmark_return=_G_BENCH)
    return {
        **params,
        "trades": stats["总交易"],
        "ret": stats["总收益率 (%)"],
        "excess": stats["超额收益 (%)"],
        "maxdd": stats["最大回撤 (%)"],
        "sharpe": stats["夏普比率"],
        "robust_sharpe": stats["稳健夏普比率"],
        "cagr": stats["CAGR (%)"],
        "mar": stats["MAR比率"],
        "final_capital": stats["最终资金 (¥)"],
    }


def _run_pm1_baseline(df: pd.DataFrame, bench: float, ts_code: str) -> dict:
    """用当前 PM1 最优参数跑一遍同标的，作为对照基线"""
    common_config.init_config(ts_code)
    base = dict(common_config.BASE_KWARGS)
    opt = dict(common_config.OPTIMAL_PARAMS_ACTIVE)
    kw = {**base, **opt, "end_date": "20260812"}
    cfg = BacktestConfig(**kw)
    df_ind = compute_indicators(df.copy(),
        entry_period=cfg.entry_period, exit_period=cfg.exit_period,
        atr_period=cfg.atr_period, ema_fast=cfg.ema_fast, ema_slow=cfg.ema_slow,
        channel_type=cfg.channel_type, bb_period=cfg.bb_period, bb_k=cfg.bb_k,
        am_ma_period=cfg.am_ma_period, mode=cfg.mode,
        pm_bb_period=cfg.pm_bb_period, pm_bb_k=cfg.pm_bb_k,
        pm_bb_width_lookback=cfg.pm_bb_width_lookback)
    trades, equity, eq_dates = run_backtest(df_ind, cfg)
    stats = calc_stats(trades, cfg.initial_capital, equity, eq_dates,
                       benchmark_return=bench)
    return {
        "trades": stats["总交易"],
        "ret": stats["总收益率 (%)"],
        "excess": stats["超额收益 (%)"],
        "maxdd": stats["最大回撤 (%)"],
        "sharpe": stats["夏普比率"],
        "robust_sharpe": stats["稳健夏普比率"],
        "cagr": stats["CAGR (%)"],
        "mar": stats["MAR比率"],
        "final_capital": stats["最终资金 (¥)"],
    }, opt


def _print_row(label: str, r: dict):
    print(f"{label:<22s} 收益率={r['ret']:8.2f}% 超额={r['excess']:8.2f}% "
          f"CAGR={r['cagr']:6.2f}% MaxDD={r['maxdd']:7.2f}% "
          f"夏普={r['sharpe']:5.2f} 稳健夏普={r['robust_sharpe']:5.2f} "
          f"MAR={r['mar']:5.2f} 交易={r['trades']:>4d} 期末={r['final_capital']:>12,.0f}")


def main():
    parser = argparse.ArgumentParser(description="bb_rebound 参数评估")
    parser.add_argument("--ts-code", type=str, default="02015.HK",
                        help="股票代码 (默认: 02015.HK)")
    args = parser.parse_args()
    ts_code = args.ts_code

    BASE_KWARGS = _base_kwargs(ts_code)
    print(f"{'='*70}")
    print(f"布林带返弹策略参数评估 — {BASE_KWARGS['stock_name']} {ts_code}")
    print(f"{'='*70}")

    cfg0 = BacktestConfig(**BASE_KWARGS)
    df = fetch_data(ts_code=cfg0.ts_code, stock_name=cfg0.stock_name,
                    market=cfg0.market, start_date=cfg0.start_date,
                    end_date=cfg0.end_date, exchange=cfg0.exchange,
                    data_source=cfg0.data_source)
    df = df.sort_values("Date").reset_index(drop=True)

    # 买入持有基准（pm_start_date 起）
    bm_mask = df["Date"] >= pd.to_datetime(cfg0.pm_start_date)
    bench = (df["Close"].iloc[-1] / df.loc[bm_mask, "Close"].iloc[0] - 1) * 100

    print(f"\n数据: {df['Date'].min().date()} ~ {df['Date'].max().date()} "
          f"({len(df)} 行), 买入持有基准收益 = {bench:.2f}%")

    # PM1 对照基线
    print(f"\n── 对照基线 ──")
    pm1_res, pm1_params = _run_pm1_baseline(df, bench, ts_code)
    _print_row("买入持有", {"ret": bench, "excess": 0.0, "cagr": float("nan"),
                            "maxdd": float("nan"), "sharpe": float("nan"),
                            "robust_sharpe": float("nan"), "mar": float("nan"),
                            "trades": 0, "final_capital": float("nan")})
    _print_row(f"PM1最优({pm1_params['pm_grid_down']}/{pm1_params['pm_grid_up']}/"
               f"{pm1_params['pm_reduce_pct']})", pm1_res)
    pm1_maxdd = pm1_res["maxdd"]

    # 扫描网格
    combos = [dict(zip(GRID.keys(), vals))
              for vals in itertools.product(*GRID.values())]
    print(f"\n── 网格扫描: {len(combos)} 组 "
          f"(period{GRID['bb_period']} × k{GRID['bb_k']} × "
          f"add{GRID['pm_add_pct']} × reduce{GRID['pm_reduce_pct']}) ──")

    results = []
    n_workers = min(8, os.cpu_count() or 4)
    with ProcessPoolExecutor(max_workers=n_workers,
                             initializer=_init_worker,
                             initargs=(df, BASE_KWARGS, bench)) as ex:
        for done, r in enumerate(ex.map(_run_one, combos, chunksize=8), 1):
            if r is not None:
                results.append(r)
            if done % 100 == 0 or done == len(combos):
                print(f"  [{done}/{len(combos)}] ...")

    results = [r for r in results if r["trades"] > 0]
    if not results:
        print("  ❌ 无有效结果")
        return

    # 按总收益率排序
    ranked = sorted(results, key=lambda x: x["ret"], reverse=True)
    print(f"\n── 按总收益率 Top10 ──")
    for r in ranked[:10]:
        _print_row(
            f"p{r['bb_period']}/k{r['bb_k']} "
            f"add{r['pm_add_pct']}/red{r['pm_reduce_pct']}", r)

    # 稳健推荐：总收益率最高且 MaxDD 不劣于 PM1 基线
    robust = [r for r in ranked if r["maxdd"] >= pm1_maxdd]
    print(f"\n── 稳健推荐（MaxDD 不劣于 PM1 基线 {pm1_maxdd:.2f}%）──")
    if robust:
        best = robust[0]
        _print_row(
            f"p{best['bb_period']}/k{best['bb_k']} "
            f"add{best['pm_add_pct']}/red{best['pm_reduce_pct']}", best)
        print(f"\n建议配置:")
        print(f"  mode='bb_rebound', bb_period={best['bb_period']}, bb_k={best['bb_k']},")
        print(f"  pm_add_pct={best['pm_add_pct']}, pm_reduce_pct={best['pm_reduce_pct']},")
        print(f"  (pm_initial_shares={BASE_KWARGS['pm_initial_shares']}, "
              f"initial_capital={BASE_KWARGS['initial_capital']}, "
              f"pm_start_date='{BASE_KWARGS['pm_start_date']}')")
    else:
        print("  无满足条件的组合（所有组合 MaxDD 均劣于 PM1 基线）")
        best = ranked[0]
        _print_row(
            f"p{best['bb_period']}/k{best['bb_k']} "
            f"add{best['pm_add_pct']}/red{best['pm_reduce_pct']}", best)

    # 保存全量网格
    out_dir = os.path.join(PROJECT_ROOT, "result")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"bb_rebound_grid_{ts_code}.xlsx")
    pd.DataFrame(results).sort_values("ret", ascending=False).to_excel(out_path, index=False)
    print(f"\n全量网格已保存: {out_path}")


if __name__ == "__main__":
    main()
