#!/usr/bin/env python3
"""共享配置 — turtle 策略所有脚本的统一配置来源

静态配置（策略改动时手改这里）：
  STOCKS           — 股票基础回测参数
  SWEEP_RANGES      — 每标的离散 sweep 范围（粗筛）
  SWEEP_DEFS        — 单参数扫描定义（精细扫描）
  DEFAULT_PARAM_SPACE — calibrate.py LHS 采样边界
  WINDOW_BARS / STEP_BARS / WARMUP / COMBINED_SCORE_WEIGHTS / PARAM_LABELS

动态配置（由脚本自动更新，持久化到 config_state.json）：
  SCORING_CONFIG    — 每标的 weights + scoring_params（calibrate.py 更新）
  OPTIMAL_PARAMS    — 每标的最优参数（turtle_sweep.py 更新）

用法：
    from strategy.position_management_1.common_config import (
        init_config, get_scoring_weights, get_scorer,
        update_scoring_config, update_optimal_params,
        BASE_KWARGS, SWEEP_RANGES_ACTIVE, OPTIMAL_PARAMS_ACTIVE,
        STOCKS, SWEEP_RANGES, SWEEP_DEFS, OPTIMAL_PARAMS,
        SCORING_CONFIG, WINDOW_BARS, STEP_BARS, WARMUP,
        COMBINED_SCORE_WEIGHTS, PARAM_LABELS, DEFAULT_PARAM_SPACE,
        _constraint_ok,
    )

    init_config("09992.HK")  # 激活指定标的
"""

import copy
import json
import os

# ── 项目根目录 ──
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ═══════════════════════════════════════════════════════════════
# 常量
# ═══════════════════════════════════════════════════════════════

WINDOW_BARS = 252
STEP_BARS = 10 #理想和老铺是5，其他10
WARMUP = 20

COMBINED_SCORE_WEIGHTS = {
    "rolling_median": 0.65,
    "sim_weighted": 0.00,
    "full_score": 0.35,
}

PARAM_LABELS = {
    "pm_grid_down": "gd",
    "pm_grid_up": "gu",
    "pm_reduce_pct": "rp",
    "pm_add_pct": "ap",
    "pm_adx_threshold": "adx",
    "pm_agg_k": "k",
}

# ═══════════════════════════════════════════════════════════════
# 默认可行参数空间（用于 calibrate.py LHS 采样）
# ═══════════════════════════════════════════════════════════════

DEFAULT_PARAM_SPACE = {
    "pm_grid_down":      (0.04, 0.20),
    "pm_grid_up":        (0.02, 0.12),
    "pm_reduce_pct":     (0.30, 0.80),
    "pm_add_pct":        (0.20, 0.60),
    "pm_adx_threshold":  (15, 50),
    "pm_agg_k":          (0.5, 1.0),
}

# ═══════════════════════════════════════════════════════════════
# 参数约束
# ═══════════════════════════════════════════════════════════════

def _constraint_ok(params: dict) -> bool:
    """参数约束检查：gd > gu 且 rp > ap"""
    gd = params.get("pm_grid_down", 0)
    gu = params.get("pm_grid_up", 0)
    rp = params.get("pm_reduce_pct", 0)
    ap = params.get("pm_add_pct", 0)
    return gd > gu and rp > ap

# ═══════════════════════════════════════════════════════════════
# 股票基础配置（回测用）
# ═══════════════════════════════════════════════════════════════

STOCKS = {
    "09992.HK": dict(  # 泡泡玛特
        ts_code="09992.HK", stock_name="泡泡玛特", market="HK",
        mode="position_management_1", per_shares=200,
        start_date="20201211", pm_start_date="20210112",
        user_dmi=True, pm_adaptive=True, pm_adaptive_grid=True,
        pm_adaptive_interp=False, pm_rsi_enabled=False,
        pm_dynamic_position=False, ema_fast=0, ema_slow=0,
        pm_initial_shares=2, initial_capital=80000,
        data_source="tushare", atr_period=14,
        pm_rsi_oversold=31, pm_rsi_overbought=76,
        pm_bb_enabled=True, pm_bb_period=10, pm_bb_width_lookback=252,
        pm_bb_width_min=0.6, pm_bb_width_max=1.3,
    ),
    "03690.HK": dict(  # 美团
        ts_code="03690.HK", stock_name="美团", market="HK",
        mode="position_management_1", per_shares=100,
        start_date="20180920", pm_start_date="20181023",
        user_dmi=True, pm_adaptive=True, pm_adaptive_grid=True,
        pm_adaptive_interp=True, pm_rsi_enabled=True,
        pm_dynamic_position=False, ema_fast=0, ema_slow=0,
        pm_initial_shares=2, initial_capital=40000,
        data_source="tushare",
        pm_rsi_oversold=37, pm_rsi_overbought=74,
        pm_bb_enabled=True, pm_bb_period=10, pm_bb_width_lookback=126,
        pm_bb_width_min=0.75, pm_bb_width_max=1.1,pm_bb_k=1.5,
    ),
    "02015.HK": dict(  # 理想
        ts_code="02015.HK", stock_name="理想", market="HK",
        mode="position_management_1", per_shares=100,
        start_date="20210812", pm_start_date="20210909",
        user_dmi=True, pm_adaptive=False, pm_adaptive_grid=True,
        pm_adaptive_interp=False, pm_rsi_enabled=False,
        pm_dynamic_position=False, ema_fast=0, ema_slow=0,
        pm_initial_shares=25, initial_capital=40000,
        data_source="tushare",
        pm_rsi_oversold=25, pm_rsi_overbought=65,
        pm_bb_enabled=True, pm_bb_period=10, pm_bb_width_lookback=21,
        pm_bb_width_min=0.6, pm_bb_width_max=1.5, pm_bb_k=1.5,
    ),
    "06181.HK": dict(  # 老铺
        ts_code="06181.HK", stock_name="老铺", market="HK",
        mode="position_management_1", per_shares=100,
        start_date="20240628", pm_start_date="20240729",
        user_dmi=True, pm_adaptive=True, pm_adaptive_grid=True,
        pm_dynamic_position=False, ema_fast=0, ema_slow=0,
        pm_initial_shares=3, initial_capital=40000,
        data_source="tushare",
        pm_rsi_oversold=31, pm_rsi_overbought=85,
        pm_bb_enabled=True, pm_bb_period=5, pm_bb_width_lookback=126,
        pm_bb_width_min=0.6, pm_bb_width_max=1.5,pm_bb_k=1.5,
    ),
    "01810.HK": dict(  # 小米
        ts_code="01810.HK", stock_name="小米", market="HK",
        mode="position_management_1", per_shares=200,
        start_date="20180709", pm_start_date="20180806",
        user_dmi=True, pm_adaptive=True, pm_adaptive_grid=True,
        pm_adaptive_interp=False, pm_rsi_enabled=True,
        pm_dynamic_position=False, ema_fast=0, ema_slow=0,
        pm_initial_shares=6, initial_capital=30000,
        data_source="tushare",
        pm_rsi_oversold=28, pm_rsi_overbought=76,
        pm_bb_enabled=True, pm_bb_period=5, pm_bb_width_lookback=252,
        pm_bb_width_min=0.7, pm_bb_width_max=1.15,pm_bb_k=1.5,
    ),
    "00700.HK": dict(  # 腾讯
        ts_code="00700.HK", stock_name="腾讯", market="HK",
        mode="position_management_1", per_shares=100,
        start_date="20160704", pm_start_date="20160801",
        user_dmi=True, pm_adaptive=True, pm_adaptive_grid=True,
        pm_adaptive_interp=True, pm_rsi_enabled=True,
        pm_dynamic_position=False, ema_fast=0, ema_slow=0,
        pm_initial_shares=0, initial_capital=80000,
        data_source="tushare",
        pm_rsi_oversold=34, pm_rsi_overbought=76,
        # BB 自适应关闭（2026-08 实测：开 BB 收益+11.3pp，但 MaxDD -16.9%→-22.9%、
        # 稳健夏普 0.89→0.79、MAR 0.74→0.56，风险调整后不划算）
        pm_bb_enabled=False, pm_bb_period=20, pm_bb_width_lookback=63,
        pm_bb_width_min=0.65, pm_bb_width_max=1.3,pm_bb_k=1.5,
    ),
    "000625.SZ": dict(  # 长安
        ts_code="000625.SZ", stock_name="长安", market="A",
        mode="position_management_1", per_shares=100,
        start_date="20160701", pm_start_date="20160729",
        user_dmi=True, pm_adaptive=False, pm_adaptive_grid=True,
        pm_dynamic_position=False, ema_fast=0, ema_slow=0,
        pm_initial_shares=9, initial_capital=40000,
        data_source="tushare",
        pm_rsi_oversold=30, pm_rsi_overbought=65,
        pm_bb_enabled=True, pm_bb_period=20, pm_bb_width_lookback=63,
        pm_bb_width_min=0.75, pm_bb_width_max=1.05,pm_bb_k=1.5,
    ),
}

# ═══════════════════════════════════════════════════════════════
# 每标的 sweep 离散搜索范围（用于 turtle_scoring.py 粗筛）
# ═══════════════════════════════════════════════════════════════

SWEEP_RANGES = {
    "09992.HK": {
        "pm_grid_down": [0.05, 0.07, 0.10, 0.12],
        "pm_grid_up": [0.03, 0.04, 0.05, 0.07],
        "pm_reduce_pct": [0.35, 0.50, 0.60, 0.75],
        "pm_add_pct": [0.25, 0.30, 0.35, 0.40, 0.50],
        "pm_adx_threshold": [25, 30, 35],
        "pm_agg_k": [1.0],
    },
    "03690.HK": {
        "pm_grid_down": [0.05, 0.07, 0.10, 0.12],
        "pm_grid_up": [0.03, 0.04, 0.05, 0.07],
        "pm_reduce_pct": [0.35, 0.50, 0.60, 0.75],
        "pm_add_pct": [0.25, 0.30, 0.35, 0.40, 0.50],
        "pm_adx_threshold": [25, 30, 35],
        "pm_agg_k": [0.5, 0.6, 0.7, 0.8, 0.9,1.0],
    },
    "02015.HK": {
        "pm_grid_down": [0.05, 0.07, 0.10, 0.12],
        "pm_grid_up": [0.03, 0.04, 0.05, 0.07],
        "pm_reduce_pct": [0.35, 0.50, 0.60, 0.75],
        "pm_add_pct": [0.25, 0.30, 0.35, 0.40, 0.50],
        "pm_adx_threshold": [25, 30, 35],
        "pm_agg_k": [1.0],
    },
    "01810.HK": {
        "pm_grid_down": [0.05, 0.07, 0.10, 0.12],
        "pm_grid_up": [0.03, 0.04, 0.05, 0.07],
        "pm_reduce_pct": [0.35, 0.50, 0.60, 0.75],
        "pm_add_pct": [0.25, 0.30, 0.35, 0.40, 0.50],
        "pm_adx_threshold": [25, 30, 35],
        "pm_agg_k": [1.0],
    },
    "06181.HK": {
        "pm_grid_down": [0.05, 0.07, 0.10, 0.12],
        "pm_grid_up": [0.03, 0.04, 0.05, 0.07],
        "pm_reduce_pct": [0.35, 0.50, 0.60, 0.75],
        "pm_add_pct": [0.25, 0.30, 0.35, 0.40, 0.50],
        "pm_adx_threshold": [25, 30, 35],
        "pm_agg_k": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    },
    "000625.SZ": {
        "pm_grid_down": [0.05, 0.07, 0.10, 0.12],
        "pm_grid_up": [0.03, 0.04, 0.05, 0.07],
        "pm_reduce_pct": [0.35, 0.50, 0.60, 0.75],
        "pm_add_pct": [0.25, 0.30, 0.35, 0.40, 0.50],
        "pm_adx_threshold": [25, 30, 35],
        "pm_agg_k": [0.5, 0.6, 0.7, 0.8, 0.9,1.0],
    },
    "00700.HK": {
            "pm_grid_down": [0.05, 0.07, 0.10, 0.12],
            "pm_grid_up": [0.03, 0.04, 0.05, 0.07],
            "pm_reduce_pct": [0.35, 0.50, 0.60, 0.75],
            "pm_add_pct": [0.25, 0.30, 0.35, 0.40, 0.50],
            "pm_adx_threshold": [25, 30, 35],
            "pm_agg_k": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    },
}

# ═══════════════════════════════════════════════════════════════
# 单参数扫描定义（用于 turtle_sweep.py 精细扫描）
# ═══════════════════════════════════════════════════════════════

SWEEP_DEFS = {
    "pm_grid_down": {
        "values": [0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075,
                   0.08, 0.085, 0.09, 0.095, 0.1, 0.105, 0.11, 0.115,
                   0.12, 0.13, 0.14, 0.15],
        "label": PARAM_LABELS["pm_grid_down"],
    },
    "pm_grid_up": {
        "values": [0.025, 0.026, 0.027, 0.028, 0.029, 0.03, 0.031, 0.032,
                   0.033, 0.034, 0.035, 0.036, 0.037, 0.038, 0.039, 0.04,
                   0.041, 0.042, 0.043, 0.044, 0.045, 0.046, 0.047, 0.048,
                   0.049, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08,
                   0.085, 0.09, 0.095, 0.10, 0.105, 0.11],
        "label": PARAM_LABELS["pm_grid_up"],
    },
    "pm_reduce_pct": {
        "values": [0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39,
                   0.4, 0.45, 0.46, 0.47, 0.48, 0.49, 0.50, 0.51, 0.52,
                   0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.60, 0.61,
                   0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.70,
                   0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.80],
        "label": PARAM_LABELS["pm_reduce_pct"],
    },
    "pm_add_pct": {
        "values": [0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28,
                   0.29, 0.30, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37,
                   0.38, 0.39, 0.40, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46,
                   0.47, 0.48, 0.49, 0.50, 0.51, 0.52, 0.53, 0.54, 0.55],
        "label": PARAM_LABELS["pm_add_pct"],
    },
    "pm_adx_threshold": {
        "values": [21, 22, 23, 24, 25, 26, 27,
                   28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                   41, 42, 43, 44, 45],
        "label": PARAM_LABELS["pm_adx_threshold"],
    },
    "pm_agg_k": {
        "values": [0.60, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68,
                   0.69, 0.70, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77,
                   0.78, 0.79, 0.80, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86,
                   0.87, 0.88, 0.89, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95,
                   0.96, 0.97, 0.98, 0.99, 1.0],
        "label": PARAM_LABELS["pm_agg_k"],
    },
}

# ═══════════════════════════════════════════════════════════════
# 默认评分配置（会被 calibrate.py 更新）
# ═══════════════════════════════════════════════════════════════

_SCORING_CONFIG_DEFAULT = {
    "02015.HK": dict(  # 理想
        weights=dict(excess=0.387,  robust_sharpe=0.321,  r_cubed=0.293),
        scoring_params=dict(
            excess=dict(type="linear_quantile", slope=1.89, intercept=-78.73),
            robust_sharpe=dict(type="linear_quantile", slope=183.67, intercept=25.20),
            r_cubed=dict(type="linear_quantile", slope=11.04, intercept=6.66),
        ),
    ),
    "09992.HK": dict(  # 泡泡玛特
        weights=dict(excess=0.376,  robust_sharpe=0.340,  r_cubed=0.285),
        scoring_params=dict(
            excess=dict(type="linear_quantile", slope=0.22, intercept=39.17),
            robust_sharpe=dict(type="linear_quantile", slope=63.83, intercept=11.38),
            r_cubed=dict(type="linear_quantile", slope=19.52, intercept=8.32),
        ),
    ),
    "03690.HK": dict(  # 美团
        weights=dict(excess=0.332,  robust_sharpe=0.407,  r_cubed=0.261),
        scoring_params=dict(
            excess=dict(type="linear_quantile", slope=0.20, intercept=3.05),
            robust_sharpe=dict(type="linear_quantile", slope=78.95, intercept=-10.0),
            r_cubed=dict(type="linear_quantile", slope=42.21, intercept=-5.13),
        ),
    ),
    "01810.HK": dict(  # 小米
        weights=dict(excess=0.404,  robust_sharpe=0.411,  r_cubed=0.185),
        scoring_params=dict(
            excess=dict(type="linear_quantile", slope=0.60, intercept=56.41),
            robust_sharpe=dict(type="linear_quantile", slope=155.17, intercept=5.0),
            r_cubed=dict(type="linear_quantile", slope=225.0, intercept=5.0),
        ),
    ),
    "06181.HK": dict(  # 老铺
        weights=dict(excess=0.392,  robust_sharpe=0.323,  r_cubed=0.285),
        scoring_params=dict(
            excess=dict(type="linear_quantile", slope=1.46, intercept=450.43),
            robust_sharpe=dict(type="linear_quantile", slope=118.42, intercept=-31.71),
            r_cubed=dict(type="linear_quantile", slope=5.07, intercept=-11.78),
        ),
    ),
    "000625.SZ": dict(  # 长安
        weights=dict(excess=0.3764, robust_sharpe=0.3358, r_cubed=0.2878),
        scoring_params=dict(
            excess=dict(type="linear_quantile", slope=1.0949, intercept=23.8280),
            robust_sharpe=dict(type="linear_quantile", slope=169.8113, intercept=79.7170),
            r_cubed=dict(type="linear_quantile", slope=204.5455, intercept=74.5455),
        ),
    ),
}

# ═══════════════════════════════════════════════════════════════
# 默认最优参数（会被 turtle_sweep.py 更新）
# ═══════════════════════════════════════════════════════════════

_OPTIMAL_PARAMS_DEFAULT = {
    "09992.HK": dict(  # 泡泡玛特
        pm_grid_down=0.15, pm_grid_up=0.038, pm_reduce_pct=0.79,
        pm_add_pct=0.50, pm_adx_threshold=21, pm_agg_k=1,
    ),
    "03690.HK": dict(  # 美团
        pm_grid_down=0.105, pm_grid_up=0.039, pm_reduce_pct=0.52,
        pm_add_pct=0.5, pm_adx_threshold=35, pm_agg_k=1,
    ),
    "02015.HK": dict(  # 理想
        pm_grid_down=0.12, pm_grid_up=0.027, pm_reduce_pct=0.51,
        pm_add_pct=0.48, pm_adx_threshold=36, pm_agg_k=1,
    ),
    "06181.HK": dict(  # 老铺
        pm_grid_down=0.075, pm_grid_up=0.05, pm_reduce_pct=0.75,
        pm_add_pct=0.50, pm_adx_threshold=36, pm_agg_k=0.9,
    ),
    "01810.HK": dict(  # 小米
        pm_grid_down=0.07, pm_grid_up=0.05, pm_reduce_pct=0.56,
        pm_add_pct=0.33, pm_adx_threshold=27, pm_agg_k=1,
    ),
    "000625.SZ": dict(  # 长安
        pm_grid_down=0.08, pm_grid_up=0.028, pm_reduce_pct=0.66,
        pm_add_pct=0.33, pm_adx_threshold=20, pm_agg_k=1,
    ),
    "00700.HK": dict(  # 腾讯
            pm_grid_down=0.07, pm_grid_up=0.05, pm_reduce_pct=0.56,
            pm_add_pct=0.33, pm_adx_threshold=27, pm_agg_k=1,
        ),
}

# ═══════════════════════════════════════════════════════════════
# 加载持久化状态
# ═══════════════════════════════════════════════════════════════

_config_dir = os.path.dirname(os.path.abspath(__file__))
CONFIG_STATE_PATH = os.path.join(_config_dir, "config_state.json")


def _load_state() -> dict:
    """加载持久化的动态配置（如果存在）。"""
    if os.path.exists(CONFIG_STATE_PATH):
        try:
            with open(CONFIG_STATE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def _save_state(state: dict):
    """保存动态配置到 JSON 文件。"""
    with open(CONFIG_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


_state = _load_state()

# 策略参数键名（用于过滤污染数据）
_PARAM_KEYS = {"pm_grid_down", "pm_grid_up", "pm_reduce_pct",
               "pm_add_pct", "pm_adx_threshold", "pm_agg_k",
               "pm_rsi_enabled", "pm_rsi_period",
               "pm_rsi_oversold", "pm_rsi_overbought"}

# 用持久化状态覆盖默认值（过滤只保留策略参数键）
OPTIMAL_PARAMS = copy.deepcopy(_OPTIMAL_PARAMS_DEFAULT)
if "optimal_params" in _state:
    for ts_code, params in _state["optimal_params"].items():
        clean = {k: v for k, v in params.items() if k in _PARAM_KEYS}
        if ts_code in OPTIMAL_PARAMS:
            OPTIMAL_PARAMS[ts_code].update(clean)
        else:
            OPTIMAL_PARAMS[ts_code] = clean

SCORING_CONFIG = copy.deepcopy(_SCORING_CONFIG_DEFAULT)
if "scoring_config" in _state:
    for ts_code, cfg in _state["scoring_config"].items():
        SCORING_CONFIG[ts_code] = cfg

# ═══════════════════════════════════════════════════════════════
# 运行时激活的配置（由 init_config() 设置）
# ═══════════════════════════════════════════════════════════════

BASE_KWARGS = None           # 当前标的的基础回测参数
SWEEP_RANGES_ACTIVE = None   # 当前标的的 sweep 离散范围
OPTIMAL_PARAMS_ACTIVE = None # 当前标的的最优参数


def init_config(ts_code: str):
    """根据 ts_code 激活对应的配置。

    设置全局变量 BASE_KWARGS, SWEEP_RANGES_ACTIVE, OPTIMAL_PARAMS_ACTIVE。
    """
    global BASE_KWARGS, SWEEP_RANGES_ACTIVE, OPTIMAL_PARAMS_ACTIVE
    if ts_code not in STOCKS:
        raise KeyError(f"未配置 {ts_code} 的 STOCKS，请在 common_config.STOCKS 中添加")
    if ts_code not in SWEEP_RANGES:
        raise KeyError(f"未配置 {ts_code} 的 SWEEP_RANGES，请在 common_config.SWEEP_RANGES 中添加")
    if ts_code not in OPTIMAL_PARAMS:
        raise KeyError(f"未配置 {ts_code} 的 OPTIMAL_PARAMS，请在 common_config.OPTIMAL_PARAMS 中添加")
    BASE_KWARGS = STOCKS[ts_code]
    SWEEP_RANGES_ACTIVE = SWEEP_RANGES[ts_code]
    OPTIMAL_PARAMS_ACTIVE = OPTIMAL_PARAMS[ts_code]


def get_scoring_weights(ts_code: str) -> dict:
    """获取指定标的的评分权重。"""
    if ts_code not in SCORING_CONFIG:
        raise KeyError(f"未配置 {ts_code} 的评分配置，请在 common_config.SCORING_CONFIG 中添加")
    return SCORING_CONFIG[ts_code]["weights"]


def get_scorer(ts_code: str):
    """获取指定标的的 AbsoluteScorer 实例。"""
    from core.scoring import AbsoluteScorer
    if ts_code not in SCORING_CONFIG:
        raise KeyError(f"未配置 {ts_code} 的评分配置，请在 common_config.SCORING_CONFIG 中添加")
    return AbsoluteScorer(SCORING_CONFIG[ts_code])


def update_scoring_config(ts_code: str, calibration: dict):
    """用校准结果更新评分配置（内存 + 持久化到 config_state.json）。

    calibration 应包含 'weights' 和 'scoring_params' 键
    （与 calibrate.py 输出的 JSON 格式一致）。
    """
    weights = calibration.get("weights", {})
    scoring_params = calibration.get("scoring_params", {})

    SCORING_CONFIG[ts_code] = {
        "weights": dict(weights),
        "scoring_params": {k: dict(v) for k, v in scoring_params.items()},
    }

    # 持久化
    _state["scoring_config"] = {
        k: v for k, v in SCORING_CONFIG.items()
    }
    _save_state(_state)
    print(f"  📝 评分配置已更新到 {CONFIG_STATE_PATH}")


def update_optimal_params(ts_code: str, params: dict):
    """更新最优参数（内存 + 持久化到 config_state.json）。

    仅保留 6 个策略参数键，自动过滤评分结果等无关字段。
    """
    _PARAM_KEYS = {"pm_grid_down", "pm_grid_up", "pm_reduce_pct",
                   "pm_add_pct", "pm_adx_threshold", "pm_agg_k"}
    filtered = {k: v for k, v in params.items() if k in _PARAM_KEYS}
    if len(filtered) < len(_PARAM_KEYS):
        missing = _PARAM_KEYS - set(filtered.keys())
        print(f"  ⚠️ update_optimal_params: 缺少参数键 {missing}，将用旧值补全")
        old = OPTIMAL_PARAMS.get(ts_code, {})
        for k in missing:
            if k in old:
                filtered[k] = old[k]

    OPTIMAL_PARAMS[ts_code] = filtered
    if OPTIMAL_PARAMS_ACTIVE is not None and isinstance(OPTIMAL_PARAMS_ACTIVE, dict):
        OPTIMAL_PARAMS_ACTIVE.clear()
        OPTIMAL_PARAMS_ACTIVE.update(filtered)

    # 持久化
    _state["optimal_params"] = {
        k: v for k, v in OPTIMAL_PARAMS.items()
    }
    _save_state(_state)
    print(f"  📝 最优参数已更新到 {CONFIG_STATE_PATH}")
