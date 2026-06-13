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
        )

        # 3. 运行回测
        trades, equity = run_backtest(df, config)

        # 4. 统计
        stats = calc_stats(trades, config.initial_capital, equity)

        # 5. 输出报告
        print_summary(stats, config)
        save_excel_report(df, trades, stats, config)

    except Exception as e:
        print(f"❌ 错误：{e}")
        import traceback
        traceback.print_exc()


# ==================== 配置示例（直接运行此脚本时使用） ====================
if __name__ == "__main__":
    # config = BacktestConfig(
    #     # ========== 股票信息 ==========
    #     ts_code="601138.SH",
    #     stock_name="工业富联",
    #     market="A",          # "A" or "HK"
    #     # ========== 回测周期 ==========
    #     years=8,
    #     # ========== 策略模式 ==========
    #     mode="long_more",         # "long" | "long_more" | "both"
    # )
    # ================ 泡泡玛特 ================
    # config = BacktestConfig(
    #     ts_code="09992.HK",
    #     stock_name="泡泡玛特",
    #     market="HK",
    #     mode="long_more",
    #     years=8,
    #     per_shares=200,
    #     max_pyramid=5,
    #     ema_fast=0,
    #     ema_slow=0,
    #     skip_first_add = True
    # )
    # config = BacktestConfig(
    #     ts_code="300274.SZ",
    #     stock_name="阳光电源",
    #     market="A",
    #     mode="long_more",
    #     max_pyramid=5,
    #     ema_slow=100,
    #     ema_fast=50,
    # )

    config = BacktestConfig(
        ts_code="601318.SH",
        stock_name="中国平安",
        ema_fast=20,
        ema_slow=250,
    )
    # config = BacktestConfig(
    #     ts_code="300059.SZ",
    #     stock_name="东方财富",
    #     ema_fast=20,
    #     ema_slow=50,
    # )
    # config = BacktestConfig(
    #     ts_code="300750.SZ",
    #     stock_name="宁德时代",
    #     ema_fast=20,
    #     ema_slow=200,
    # )

    # config = BacktestConfig(
    #     ts_code="002475.SZ",
    #     stock_name="立讯精密",
    #     ema_fast=20,
    #     ema_slow=200,
    # )
    # config = BacktestConfig(
    #     ts_code="601012.SH",
    #     stock_name="隆基绿能",
    #     market="A",
    #     mode="long_more",
    #     max_pyramid=5,
    #     ema_slow=100,
    #     ema_fast=50,
    # )
    # config = BacktestConfig(
    #     ts_code="00388.HK",
    #     stock_name="香港交易所",
    #     market="HK",
    #     ema_fast=30,
    #     ema_slow=60,
    # )
    # config = BacktestConfig(
    #     ts_code="03690.HK",
    #     stock_name="美团",  # 2018-09-20开市
    #     market="HK",
    #     ema_fast=20,
    #     ema_slow=60,
    # )
    # config = BacktestConfig(
    #     ts_code="03690.HK",
    #     stock_name="美团",  # 2018-09-20开市
    #     market="HK",
    #     mode="long_more",
    #     max_pyramid=5,
    #     channel_type="bollinger"
    # )
    # config = BacktestConfig(
    #     ts_code="02015.HK",
    #     stock_name="理想",  # 2018-09-20开市
    #     market="HK",
    #     mode="short_more",
    #     ema_fast=20,
    #     ema_slow=80,
    # )
    # config = BacktestConfig(
    #     ts_code="06181.HK",
    #     stock_name="老铺",  # 2018-09-20开市
    #     market="HK",
    #     mode="short_more",
    #     ema_fast=0,
    #     ema_slow=0,
    #     initial_capital=1200000,
    # )
    # config = BacktestConfig(
    #     ts_code="09992.HK",
    #     stock_name="泡泡玛特",
    #     market="HK",
    #     mode="long",
    #     years=5,
    #     per_shares=200,
    #     max_pyramid=5,
    #     channel_type="bollinger",
    #     bb_period=10,
    #     bb_k=1,
    # )
    # config = BacktestConfig(
    #     ts_code="00700.HK",
    #     stock_name="腾讯",
    #     market="HK",
    #     mode="long_more",
    #     initial_capital=200000.0,
    #     max_pyramid=5,
    #     channel_type="bollinger",
    #     bb_period=10,
    #     bb_k=1,
    # )
    # config = BacktestConfig(
    #     ts_code="300750.SZ",
    #     stock_name="宁德时代",
    #     market="A",
    #     mode="long_more",
    #     max_pyramid=5,
    #     channel_type="bollinger",
    #     bb_period=10,
    #     bb_k=1,
    # )
    # config = BacktestConfig(
    #     ts_code="300502.SZ",
    #     stock_name="新易盛",
    #     market="A",
    #     mode="long_more",
    #     max_pyramid=5,
    #     channel_type="bollinger",
    #     bb_period=10,
    #     bb_k=1,
    # )
    # config = BacktestConfig(
    #     ts_code="300502.SZ",
    #     stock_name="新易盛",
    #     market="A",
    #     mode="long_more",
    #     max_pyramid=5,
    #     ema_fast=50,
    #     channel_type="donchian",
    #     bb_period=10,
    #     bb_k=1,
    # )

    # config = BacktestConfig(
    #     ts_code="01810.HK",
    #     stock_name="小米",  # 2018-07-09开市
    #     market="HK",
    #     ema_fast=20,
    #     ema_slow=200,
    # )
    # config = BacktestConfig(
    #     ts_code="00939.HK",
    #     stock_name="建设银行",  # 2018-07-09开市
    #     market="HK",
    #     ema_fast=0,
    #     ema_slow=0,
    # )

    # config = BacktestConfig(
    #     ts_code="00148.HK",
    #     stock_name="建滔集团",  # 2018-07-09开市
    #     market="HK",
    #     ema_fast=25,
    #     ema_slow=60,
    # )
    config = BacktestConfig(
        ts_code="513500.SH",
        stock_name="标普500",  # 2018-07-09开市
        market="ETF",
        ema_fast=0,
        ema_slow=0,
    )
    run(config)
