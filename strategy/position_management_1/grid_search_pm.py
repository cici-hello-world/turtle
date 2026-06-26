#!/usr/bin/env python3
"""临时脚本：仓位管理参数网格搜索，找总收益率最大的参数组合

搜索 pm_grid_up, pm_grid_down, pm_add_pct, pm_reduce_pct 四个参数，
其他参数保持不变（沿用 run_backtest.py 中当前活跃配置）。
"""

import itertools
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.config import BacktestConfig
from core.data import fetch_data
from core.indicators import compute_indicators
from core.engine import run_backtest
from core.statistics import calc_stats


def grid_search():
    # ==================== 固定参数（当前 run_backtest.py 活跃配置） ====================
    base_kwargs = dict(
        ts_code="01810.HK",
        stock_name="小米",
        market="HK",
        mode="position_management_1",
        per_shares=100,
        pm_start_date="20251001",
        start_date="20180709",
        user_dmi =True,
        ema_fast=0,
        ema_slow=0,
        pm_initial_shares=8,
        initial_capital=40000,
        data_source="tushare",
    )

    # ==================== 搜索空间 ====================
    pm_grid_up_values    = [0.02, 0.03, 0.05, 0.07, 0.10]
    pm_grid_down_values  = [0.02, 0.03, 0.05, 0.07, 0.10]
    pm_add_pct_values    = [0.03, 0.05, 0.065, 0.10, 0.15]
    pm_reduce_pct_values = [0.25, 0.50, 0.75, 1.0]

    # ==================== 预加载数据（数据只取一次，指标也只算一次） ====================
    print("📥 获取数据 & 计算指标 ...")
    config0 = BacktestConfig(**base_kwargs)
    df_base = fetch_data(
        ts_code=config0.ts_code,
        stock_name=config0.stock_name,
        market=config0.market,
        start_date=config0.start_date,
        end_date=config0.end_date,
        exchange=config0.exchange,
        data_source=config0.data_source,
    )
    df_base = compute_indicators(
        df_base,
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
    print(f"   数据行数: {len(df_base)}")

    # 买入持有基准收益（从仓位管理开始日 pm_start_date 算起）
    df_pm = df_base[df_base["Date"] >= config0.pm_start_date]
    if len(df_pm) >= 2:
        buy_hold_return = ((df_pm["Close"].iloc[-1] - df_pm["Close"].iloc[0])
                           / df_pm["Close"].iloc[0] * 100)
    else:
        buy_hold_return = 0.0
    print(f"   买入持有收益率 (自{config0.pm_start_date}): {buy_hold_return:+.2f}%\n")

    # ==================== 网格搜索 ====================
    best_result = None
    best_stats = None
    best_return = -float("inf")

    total = (len(pm_grid_up_values) * len(pm_grid_down_values) *
             len(pm_add_pct_values) * len(pm_reduce_pct_values))
    count = 0

    for up, down, add, reduce in itertools.product(
        pm_grid_up_values, pm_grid_down_values, pm_add_pct_values, pm_reduce_pct_values
    ):
        count += 1
        config = BacktestConfig(
            **base_kwargs,
            pm_grid_up=up,
            pm_grid_down=down,
            pm_add_pct=add,
            pm_reduce_pct=reduce,
        )

        try:
            trades, equity = run_backtest(df_base.copy(), config)
            stats = calc_stats(trades, config.initial_capital, equity)
            total_return = stats["总收益率 (%)"]
            excess = total_return - buy_hold_return  # 超额收益

            marker = ""
            if total_return > best_return:
                best_return = total_return
                best_result = (up, down, add, reduce)
                best_stats = stats
                marker = " ⭐"

            print(f"[{count:3d}/{total}] "
                  f"up={up:.2f} down={down:.2f} add={add:.3f} reduce={reduce:.2f} "
                  f"→ 策略={total_return:7.2f}%  超额={excess:+7.2f}%{marker}")

        except Exception as e:
            print(f"[{count:3d}/{total}] "
                  f"up={up:.2f} down={down:.2f} add={add:.3f} reduce={reduce:.2f} "
                  f"→ ❌ {e}")

    # ==================== 输出最优 ====================
    print("\n" + "=" * 60)
    print("🏆 最优参数组合")
    print("=" * 60)
    if best_result is None:
        print("❌ 没有找到有效结果")
        return

    up, down, add, reduce = best_result
    best_excess = best_return - buy_hold_return
    print(f"   pm_grid_up    = {up:.2f}")
    print(f"   pm_grid_down  = {down:.2f}")
    print(f"   pm_add_pct    = {add:.3f}")
    print(f"   pm_reduce_pct = {reduce:.2f}")
    print("-" * 60)
    print(f"   买入持有收益率  = {buy_hold_return:+.2f}%")
    print(f"   策略总收益率    = {best_return:.2f}%")
    print(f"   超额收益       = {best_excess:+.2f}%")
    if best_stats:
        print(f"   总交易次数     = {best_stats['总交易']}")
        print(f"   胜率           = {best_stats['胜率 (%)']}%")
        print(f"   盈利次数       = {best_stats['盈利次数']}")
        print(f"   亏损次数       = {best_stats['亏损次数']}")
        print(f"   最大回撤       = {best_stats['最大回撤 (%)']}%")
        print(f"   最终资金       = ¥{best_stats['最终资金 (¥)']:,.2f}")
        print(f"   总盈亏         = ¥{best_stats['总盈亏 (¥)']:,.2f}")
        print(f"   夏普比率       = {best_stats['夏普比率']}")
        print(f"   CAGR           = {best_stats['CAGR (%)']}%")
        print(f"   稳健夏普比率   = {best_stats['稳健夏普比率']}")
        print(f"   R立方          = {best_stats['R立方']}")
    print("=" * 60)


if __name__ == "__main__":
    grid_search()
