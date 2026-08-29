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
            mode = config.mode,
            pm_bb_period=config.pm_bb_period,
            pm_bb_k=config.pm_bb_k,
            pm_bb_width_lookback=config.pm_bb_width_lookback,
            pm_rsi_period=config.pm_rsi_period,
        )

        # 3. 运行回测
        trades, equity, eq_dates = run_backtest(df, config)

        # 4. 基准收益（同期买入持有）
        bm_start_str = config.pm_start_date or config.don_start_date
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
    # long_more 模式示例
    ("09992.HK", "long_more"): dict(
        ts_code="09992.HK", stock_name="泡泡玛特", market="HK",
        mode="long_more", years=5, per_shares=200,
        initial_capital=400000, max_pyramid=5,don_start_date="20260629",
        ema_fast=0, ema_slow=0, data_source="akshare",
    ),
    ("03690.HK", "long_more"): dict(
        ts_code="03690.HK", stock_name="美团", market="HK",
        mode="long_more",ema_fast=0, ema_slow=0,user_dmi=False,
        start_date="20180920",don_start_date="20260629",initial_capital=40000,
        data_source="akshare",
    ),
    ("06181.HK", "long_more"): dict(
        ts_code="06181.HK", stock_name="老铺", market="HK",
        mode="long_more", ema_fast=0, ema_slow=0, user_dmi=False,
        start_date="20240628", don_start_date="20240729",
        initial_capital=200000,
    ),
    ("00700.HK", "long_more"): dict(
        ts_code="00700.HK", stock_name="腾讯", market="HK",
        mode="long_more", ema_fast=0, ema_slow=0, user_dmi=False,
        start_date="20160704",don_start_date="20160801",
        initial_capital=200000,
    ),
    ("01810.HK", "long_more"): dict(
        ts_code="01810.HK", stock_name="小米", market="HK",
        mode="long_more",ema_fast=0, ema_slow=0,user_dmi=False,
        start_date="20180709",don_start_date="20180806", initial_capital=40000,
        data_source="akshare",
    ),
    ("601138.SH", "long_more"): dict(
            ts_code="601138.SH", stock_name="工业富联", market="A",
            mode="long_more",ema_fast=25, ema_slow=200,user_dmi=False,
            start_date="20180608",don_start_date="20180706", initial_capital=40000,
            data_source="akshare",
    ),
    # ==================== dual_ma 双均线模式（聚宽经典版） ====================
    # 参数为 2026-08 扫描最优（按稳健夏普），收益/回撤见各条目注释
    ("09992.HK", "dual_ma"): dict(
        ts_code="09992.HK", stock_name="泡泡玛特", market="HK",
        mode="dual_ma", per_shares=200, start_date="20201211",
        pm_start_date="20210112", initial_capital=80000, data_source="tushare",
        dm_fast=20, dm_slow=120,  # 584.6%/-42.9%/1.44（收益与稳健夏普双最优）
    ),
    ("03690.HK", "dual_ma"): dict(
        ts_code="03690.HK", stock_name="美团", market="HK",
        mode="dual_ma", per_shares=100, start_date="20180920",
        pm_start_date="20181023", initial_capital=40000, data_source="tushare",
        dm_fast=10, dm_slow=60,  # 193.8%/-55.0%/0.57（稳健最优；5/60 收益255.6%但夏普0.44）
    ),
    ("06181.HK", "dual_ma"): dict(
        ts_code="06181.HK", stock_name="老铺", market="HK",
        mode="dual_ma", per_shares=100, start_date="20240628",
        pm_start_date="20240729", initial_capital=40000, data_source="tushare",
        dm_fast=5, dm_slow=20,  # 156.1%/-65.6%/0.37（慢线组合全程无金叉，仅此组合有效）
    ),
    ("01810.HK", "dual_ma"): dict(
        ts_code="01810.HK", stock_name="小米", market="HK",
        mode="dual_ma", per_shares=200, start_date="20180709",
        pm_start_date="20180806", initial_capital=30000, data_source="tushare",
        dm_fast=20, dm_slow=60,  # 346.0%/-44.2%/0.75（稳健最优；10/120 收益387.3%但夏普0.64）
    ),
    ("00700.HK", "dual_ma"): dict(
        ts_code="00700.HK", stock_name="腾讯", market="HK",
        mode="dual_ma", per_shares=100, start_date="20160704",
        pm_start_date="20160801", initial_capital=80000, data_source="tushare",
        dm_fast=10, dm_slow=120,  # 276.5%/-27.3%/0.62（收益与稳健夏普双最优）
    ),
    ("600977.SH", "dual_ma"): dict(
        ts_code="600977.SH", stock_name="中国电影", market="A",
        mode="dual_ma", per_shares=100, start_date="20160809",
        pm_start_date="20160901", initial_capital=40000, data_source="tushare",
        dm_fast=20, dm_slow=120,  # 全线负收益(-42.9%起)，双均线不适合该标的，参数仅供参考
    ),

}
from strategy.position_management_1 import common_config
from strategy.bb_rebound import bb_config
BASE_KWARGS = None
BEST_PARAMS = None

def get_stock_config(ts_code: str, mode: str = "position_management_1") -> dict:
    """获取指定标的和模式对应的回测配置。"""
    if mode == "position_management_1":
        global BASE_KWARGS, BEST_PARAMS
        common_config.init_config(ts_code)
        BASE_KWARGS = dict(common_config.BASE_KWARGS)  # 复制，避免污染共享配置
        BASE_KWARGS["pm_start_date"] = "20260629"
        # BASE_KWARGS["end_date"] = "20230628"
        BEST_PARAMS = dict(common_config.OPTIMAL_PARAMS_ACTIVE)
        return {**BASE_KWARGS, **BEST_PARAMS}
    if mode == "bb_rebound":
        if ts_code not in bb_config.BB_STOCKS:
            raise KeyError(f"未配置 {ts_code} 的 bb_rebound 参数，请先在 strategy/bb_rebound/bb_config.py 中添加")
        return dict(bb_config.BB_STOCKS[ts_code])  # 复制，避免污染共享配置
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
