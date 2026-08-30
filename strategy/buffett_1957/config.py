#!/usr/bin/env python3
"""巴菲特1957组合策略配置"""
from dataclasses import dataclass


@dataclass
class Buffett1957Config:
    # ── 数据范围 ──
    start_date: str = "20050101"
    end_date: str = "20260828"

    # ── 估值 → 权重映射（恒指等权静态PE历史分位） ──
    lo: float = 0.30           # 分位 ≤ lo → 全仓价值组合（低估）
    hi: float = 0.70           # 分位 ≥ hi → 价值组合最低权重（高估，其余现金）
    w_min: float = 0.25        # 高估时价值组合最低权重
    min_history: int = 120     # 分位数有效所需最少月数（不足时按 0.5 分位处理）

    # ── 价值组合（全市场价值选股） ──
    universe_n: int = 400      # 季度流动性 Top-N 股票池
    basket_n: int = 30         # 等权持仓数
    liquidity_days: int = 5    # 季度末取平均成交额的天数

    # ── 资金 ──
    initial_capital: float = 1_000_000.0   # 港元
