#!/usr/bin/env python3
"""海龟策略回测 —— 统一入口脚本

用法：
    # 直接修改下方配置后运行
    python run_backtest.py

    # 或在代码中导入使用
    from run_backtest import run
    from core.config import BacktestConfig

    config = BacktestConfig(ts_code="000625.SZ", stock_name="长安汽车", ...)
    run(config)
"""
import pandas as pd

from core.config import BacktestConfig
from core.data import fetch_data
from core.indicators import compute_indicators
from core.engine import run_backtest
from core.statistics import calc_stats
from core.reporter import print_summary, save_excel_report


def run(config: BacktestConfig):
    """执行完整回测流程"""
    try:
        # 1. 获取数据
        df = fetch_data(
            ts_code=config.ts_code,
            stock_name=config.stock_name,
            market=config.market,
            start_date=config.start_date,
            end_date=config.end_date,
            exchange=config.exchange,
            data_source=config.data_source,
        )

        # 2. 计算指标
        df = compute_indicators(
            df,
            entry_period=config.entry_period,
            exit_period=config.exit_period,
            atr_period=config.atr_period,
            ema_fast=config.ema_fast,
            ema_slow=config.ema_slow,
            channel_type=config.channel_type,
            bb_period=config.bb_period,
            bb_k=config.bb_k,
            am_ma_period=config.am_ma_period,
            mode = config.mode
        )

        # 3. 运行回测
        trades, equity, eq_dates = run_backtest(df, config)

        # 4. 基准收益（同期买入持有）
        bm_start_str = config.pm_start_date or config.start_date
        bm_mask = df["Date"] >= pd.to_datetime(bm_start_str)
        bm_start = df.loc[bm_mask, "Close"].iloc[0]
        bm_end = df["Close"].iloc[-1]
        benchmark_return = (bm_end - bm_start) / bm_start * 100

        # 5. 统计
        stats = calc_stats(trades, config.initial_capital, equity, eq_dates,
                          benchmark_return=benchmark_return)

        # 6. 输出报告
        print_summary(stats, config)
        save_excel_report(df, trades, stats, config)

    except Exception as e:
        print(f"❌ 错误：{e}")
        import traceback
        traceback.print_exc()


# ==================== 配置 — key 为 (ts_code, mode) ====================
STOCK_CONFIGS = {
    ("03690.HK", "position_management_1"): dict(
        ts_code="03690.HK", stock_name="美团", market="HK",
        mode="position_management_1",
        pm_start_date="20250625", start_date="20180920",
        user_dmi=True, pm_adaptive_grid=False,
        pm_dynamic_position=False, ema_fast=0, ema_slow=0,
        pm_grid_down=0.10, pm_grid_up=0.027, pm_reduce_pct=0.57,
        pm_add_pct=0.52, pm_adx_threshold=35, pm_agg_k=1,
        pm_initial_shares=2, initial_capital=40000,
        data_source="akshare",
    ),
    ("09992.HK", "position_management_1"): dict(
        ts_code="09992.HK", stock_name="泡泡玛特", market="HK",
        mode="position_management_1", per_shares=200,
        pm_start_date="20260629", start_date="20201211",
        user_dmi=True, pm_adaptive_grid=False,
        pm_dynamic_position=False, ema_fast=0, ema_slow=0,
        pm_grid_down=0.14, pm_grid_up=0.055, pm_reduce_pct=0.75,
        pm_add_pct=0.5, pm_adx_threshold=31, pm_agg_k=1,
        pm_initial_shares=2, initial_capital=40000,
        data_source="tushare",
    ),
    ("02015.HK", "position_management_1"): dict(
        ts_code="02015.HK", stock_name="理想", market="HK",
        mode="position_management_1",
        user_dmi=True, ema_fast=0, ema_slow=0,
        pm_start_date="20260629", start_date="20220813",
        pm_grid_down=0.05, pm_grid_up=0.04, pm_reduce_pct=0.75,
        pm_add_pct=0.50, pm_adx_threshold=24, pm_agg_k=1,
        pm_initial_shares=25, initial_capital=40000,
        pm_dynamic_position=False, pm_adaptive_grid=False,
        data_source="tushare",
    ),
    ("01810.HK", "position_management_1"): dict(
        ts_code="01810.HK", stock_name="小米", market="HK",
        mode="position_management_1",
        user_dmi=True, ema_fast=0, ema_slow=0,
        pm_start_date="20250509", start_date="20180709", end_date="20260515",
        pm_grid_down=0.07, pm_grid_up=0.05, pm_reduce_pct=0.69,
        pm_add_pct=0.33, pm_adx_threshold=26, pm_agg_k=1,
        pm_initial_shares=8, initial_capital=40000,
        pm_dynamic_position=False, pm_adaptive_grid=False,
        data_source="akshare",
    ),
    ("06181.HK", "position_management_1"): dict(
        ts_code="06181.HK", stock_name="老铺", market="HK",
        mode="position_management_1",
        pm_start_date="20260629", start_date="20240628",
        user_dmi=True, pm_adaptive_grid=True,
        pm_grid_down=0.045, pm_grid_up=0.044, pm_reduce_pct=0.46,
        pm_add_pct=0.4, pm_adx_threshold=32, pm_agg_k=0.76,
        ema_slow=0, ema_fast=0, pm_initial_shares=3,
        initial_capital=40000, pm_dynamic_position=False,
        data_source="tushare",
    ),
    ("00700.HK","position_management_1"): dict(  # 腾讯
        ts_code="00700.HK", stock_name="腾讯", market="HK",
        mode="position_management_1",
        pm_start_date="20251001",start_date="20160629",
        user_dmi=True, pm_adaptive=True, pm_adaptive_grid=False,
        pm_dynamic_position=False, ema_fast=0, ema_slow=0,
        pm_initial_shares=0, initial_capital=60000,
        data_source="tushare",
    ),
    # long_more 模式示例
    ("09992.HK", "long_more"): dict(
        ts_code="09992.HK", stock_name="泡泡玛特", market="HK",
        mode="long_more", years=5, per_shares=200,
        initial_capital=400000, max_pyramid=5,
        ema_fast=0, ema_slow=0, data_source="akshare",
    ),
    ("03690.HK", "long_more"): dict(
        ts_code="03690.HK", stock_name="美团", market="HK",
        mode="long_more", max_pyramid=5, channel_type="bollinger",
    ),
    ("06181.HK", "long_more"): dict(
        ts_code="06181.HK", stock_name="老铺", market="HK",
        mode="long_more", ema_fast=0, ema_slow=0, user_dmi=False,
        start_date="20240628", end_date="20260624",
        initial_capital=200000,
    ),
}


def get_stock_config(ts_code: str, mode: str = "position_management_1") -> dict:
    """获取指定标的和模式对应的回测配置。"""
    key = (ts_code, mode)
    if key not in STOCK_CONFIGS:
        raise KeyError(f"未配置 {ts_code} mode={mode} 的回测参数，请在 STOCK_CONFIGS 中添加")
    return STOCK_CONFIGS[key]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="海龟策略回测")
    parser.add_argument("--ts-code", type=str, default="03690.HK",
                        help="股票代码 (默认: 03690.HK)")
    parser.add_argument("--mode", type=str, default="position_management_1",
                        help="策略模式 (默认: position_management_1)")
    args = parser.parse_args()
    config = BacktestConfig(**get_stock_config(args.ts_code, args.mode))
    run(config)
