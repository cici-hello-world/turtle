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


# ==================== 配置示例（直接运行此脚本时使用） ====================
if __name__ == "__main__":
    # config = BacktestConfig(
    #     # ========== 股票信息 ==========
    #     ts_code="601138.SH",
    #     stock_name="工业富联",
    #     ema_fast=25,
    #     ema_slow=200,
    # )
    # config=BacktestConfig(
    #     # ========== 股票信息 ==========
    #     ts_code="601919.SH",
    #     stock_name="中远海控",
    #     ema_fast=30,
    #     ema_slow=150,
    # )
    # ================ 泡泡玛特 ================
    # config = BacktestConfig(
    #     ts_code="09992.HK",
    #     stock_name="泡泡玛特",
    #     market="HK",
    #     mode="long_more",
    #     years=5,
    #     per_shares=200,
    #     initial_capital=400000,
    #     max_pyramid=5,
    #     ema_fast=0,
    #     ema_slow=0,
    #     data_source="akshare",
    # )
    # config = BacktestConfig(
    #     ts_code="09992.HK",
    #     stock_name="泡泡玛特",
    #     market="HK",
    #     mode="anti_martingale",
    #     initial_capital =400000,
    #     years=5,
    #     per_shares=200,
    #     ema_fast=25,
    #     ema_slow=80,
    #     am_ma_period =50
    # )
    # config = BacktestConfig(
    #         ts_code="02513.HK",
    #         stock_name="智谱",
    #         market="HK",
    #         mode="long_more",
    #         initial_capital =4000000,
    #         years=1,
    #         atr_period=20,
    #         ema_fast=0,
    #         ema_slow=0,
    #     )
    # config = BacktestConfig(
    #     ts_code="300274.SZ",
    #     stock_name="阳光电源",
    #     ema_fast=50,
    #     ema_slow=250,
    # )

    # config = BacktestConfig(
    #     ts_code="603799.SH",
    #     stock_name="华友钴业",
    #     ema_fast=25,
    #     ema_slow=80,
    # )
    # config = BacktestConfig(
    #     ts_code="601633.SH",
    #     stock_name="长城汽车",
    #     ema_fast=20,
    #     ema_slow=200,
    # )
    # config = BacktestConfig(
    #     ts_code="000408.SZ",
    #     stock_name="藏格矿业",
    #     ema_fast=50,
    #     ema_slow=200,
    # )
    # config = BacktestConfig(
    #     ts_code="300418.SZ",
    #     stock_name="昆仑万维",
    #     ema_fast=25,
    #     ema_slow=100,
    # )
    # config = BacktestConfig(
    #     ts_code="601888.SH",
    #     stock_name="中国中免",
    #     ema_fast=30,
    #     ema_slow=50,
    # )
    # config = BacktestConfig(
    #     ts_code="601318.SH",
    #     stock_name="中国平安",
    #     ema_fast=20,
    #     ema_slow=250,
    # )
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
    #     ts_code="601788.SH",
    #     stock_name="光大证券",
    #     ema_fast=20,
    #     ema_slow=50,
    # )
    # config = BacktestConfig(
    #     ts_code="603986.SH",
    #     stock_name="兆易创新",
    #     ema_fast=0,
    #     ema_slow=0,
    #     initial_capital=400000,
    # )
    # config = BacktestConfig(
    #     ts_code="688256.SH",
    #     stock_name="寒武纪",
    #     initial_capital=400000,
    #     ema_fast=50,
    #     ema_slow=300,
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
    #     ts_code="688530.SH",
    #     stock_name="欧莱新材",
    #     market="A",
    #     mode="long_more",
    #     max_pyramid=5,
    #     ema_slow=0,
    #     ema_fast=0,
    # )
    # config = BacktestConfig(
    #     ts_code="301396.SZ",
    #     stock_name="宏景科技",
    #     market="A",
    #     mode="long_more",
    #     max_pyramid=5,
    #     ema_slow=20,
    #     ema_fast=300,
    #     initial_capital=200000,
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
    #     ts_code="09992.HK",
    #     stock_name="泡泡玛特",  # 20201211开市
    #     market="HK",
    #     mode="position_management_1",
    #     per_shares=200,
    #     pm_start_date="20251001",
    #     start_date="20201211",
    #     user_dmi=True,
    #     ema_fast=0,
    #     ema_slow=0,
    #     pm_grid_up=0.1,
    #     pm_grid_down=0.1,
    #     pm_add_pct=0.030,
    #     pm_reduce_pct=0.75,
    #     pm_initial_shares=2,   # 手数（26手 = 2600股）
    #     initial_capital=40000,
    #     data_source="akshare",
    # )
    config = BacktestConfig(
        ts_code="02015.HK",
        stock_name="理想",  # 2018-09-20开市
        market="HK",
        mode="position_management_1",
        user_dmi=True,
        ema_fast=0,
        ema_slow=0,
        pm_start_date="20251001",
        start_date="20220813",
        pm_grid_down=0.10,
        pm_grid_up = 0.03,
        pm_reduce_pct = 0.75,
        pm_add_pct = 0.03,
        pm_adx_threshold = 22,
        pm_grid_down_agg = 0.02,
        pm_grid_up_agg = 0.02,
        pm_reduce_pct_agg = 0.75,
        pm_take_profit_ratio = 1.30,
        pm_initial_shares=25,  # 手数（26手 = 2600股）
        initial_capital=40000,
        data_source="akshare",
     )
    # config = BacktestConfig(
    #     ts_code="03690.HK",
    #     stock_name="美团",  # 2018-09-20开市
    #     market="HK",
    #     mode="position_management_1",
    #     pm_start_date="20251001",
    #     start_date="20180920",
    #     user_dmi=True,
    #     ema_fast=0,
    #     ema_slow=0,
    #     pm_grid_up=0.05,
    #     pm_grid_down = 0.07,
    #     pm_add_pct = 0.15,
    #     pm_reduce_pct = 0.75,
    #     pm_initial_shares=2,  # 手数（26手 = 2600股）
    #     initial_capital=40000,
    #     data_source="akshare",
    # )
    # config = BacktestConfig(
    #     ts_code="01810.HK",
    #     stock_name="小米",
    #     market="HK",
    #     mode="position_management_1",
    #     pm_start_date="20251001",
    #     start_date="20180709",
    #     pm_grid_up=0.02,
    #     pm_grid_down = 0.05,
    #     pm_add_pct = 0.03,
    #     pm_reduce_pct = 0.75,
    #     user_dmi=True,
    #     ema_fast=0,
    #     ema_slow=0,
    #     pm_initial_shares=8,  # 手数（26手 = 2600股）
    #     initial_capital=40000,
    #     data_source="akshare",
    # )
    # ========== 老铺黄金 PM1 - 旧版参考 ==========
    # 高收益版: grid_down=0.03 grid_up=0.05 reduce=0.75 ADX=20 agg_rp=0.25 → +16.2% -17.0% 33笔
    # 低回撤版: grid_down=0.10 grid_up=0.07 reduce=0.25 ADX=25 agg_rp=0.50 → +12.4% -6.3%  21笔
    # ========== 老铺黄金 PM1 - 最优版（收益 +14.2%, 回撤 -6.2%, 23笔, 稳健夏普 2.74） ==========
    # config = BacktestConfig(
    #     ts_code="06181.HK",
    #     stock_name="老铺",
    #     market="HK",
    #     mode="position_management_1",
    #     start_date="20240628",
    #     pm_start_date="20251001",
    #     user_dmi=True,
    #     pm_grid_up=0.04,
    #     pm_grid_down=0.10,
    #     pm_add_pct=0.02,
    #     pm_reduce_pct=0.25,
    #     pm_adx_threshold=25,
    #     pm_grid_up_agg=0.03,
    #     pm_grid_down_agg=0.05,
    #     pm_reduce_pct_agg=0.50,
    #     ema_slow=0,
    #     ema_fast=0,
    #     pm_initial_shares=3,
    #     initial_capital=40000,
    #     pm_dynamic_position=True,
    #     data_source="akshare",
    # )
    # config = BacktestConfig(
    #     ts_code="06181.HK",
    #     stock_name="老铺",  # 2018-09-20开市
    #     market="HK",
    #     mode="long_more",
    #     ema_fast=0,
    #     ema_slow=0,
    #     user_dmi=False,
    #     start_date="20240628",
    #     end_date="20260624",
    #     initial_capital=200000,
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
    #     ema_fast=30,
    #     ema_slow=60,
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
    # config = BacktestConfig(
    #     ts_code="513500.SH",
    #     stock_name="标普500",  # 2018-07-09开市
    #     market="ETF",
    #     ema_fast=0,
    #     ema_slow=0,
    # )
    run(config)
