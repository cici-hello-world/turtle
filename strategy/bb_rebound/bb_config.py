#!/usr/bin/env python3
"""bb_rebound 策略配置 — 布林带返弹模式的标的与参数

参数来源：strategy/bb_rebound/evaluate_bb_rebound.py 网格评估
（理想 02015.HK，2026-08 评估：p30/k2.5 add0.50 red0.75，
收益率 174.1% / MaxDD -30.2%，优于 PM1 的 59.5% / -21.8%；
长安 000625.SZ，2026-08 评估：p10/k2.0 add0.20 red0.40，
收益率 89.0% / MaxDD -31.1%，优于 PM1 的 -13.3% / -20.0%）

用法：
    from strategy.bb_rebound.bb_config import BB_STOCKS, get_config
    config = get_config("02015.HK")
    config = get_config("02015.HK", pm_start_date="20260801")  # 每日信号用
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from core.config import BacktestConfig

BB_STOCKS = {
    "02015.HK": dict(  # 理想
        ts_code="02015.HK", stock_name="理想", market="HK",
        mode="bb_rebound", per_shares=100,
        start_date="20210812", pm_start_date="20210909",
        pm_initial_shares=25, initial_capital=40000,
        data_source="tushare",
        bb_period=30, bb_k=2.5,
        pm_add_pct=0.50, pm_reduce_pct=0.75,
        pm_max_position_pct=0.9, pm_min_cash_pct=0.10,
        pm_take_profit_ratio=100,
    ),
    "000625.SZ": dict(  # 长安
        ts_code="000625.SZ", stock_name="长安", market="A",
        mode="bb_rebound", per_shares=100,
        start_date="20160701", pm_start_date="20160729",
        pm_initial_shares=9, initial_capital=40000,
        data_source="tushare",
        bb_period=10, bb_k=2.0,
        pm_add_pct=0.20, pm_reduce_pct=0.40,
        pm_max_position_pct=0.9, pm_min_cash_pct=0.10,
        pm_take_profit_ratio=100,
    ),
}


def get_config(ts_code: str, **overrides) -> BacktestConfig:
    """构建 BacktestConfig，支持覆盖参数（如每日信号脚本覆盖 pm_start_date）。"""
    if ts_code not in BB_STOCKS:
        raise KeyError(f"未配置 {ts_code} 的 bb_rebound 参数，请在 bb_config.BB_STOCKS 中添加")
    return BacktestConfig(**{**BB_STOCKS[ts_code], **overrides})
