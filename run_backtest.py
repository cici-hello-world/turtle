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
import pandas as pd


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
            dividend_info=config.dividend_info,
        )

        # 2. 计算指标
        df = compute_indicators(
            df,
            entry_period=config.entry_period,
            exit_period=config.exit_period,
            atr_period=config.atr_period,
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
    #
    #     # ========== 回测周期 ==========
    #     years=8,
    #
    #     # ========== 港股分红（仅港股） ==========
    #     # dividend_info=pd.DataFrame({
    #     #     "ex_date": ["20260515"],
    #     #     "dividend": [2.7248],
    #     # }),
    #
    #     # ========== 策略模式 ==========
    #     mode="long",         # "long" | "long_more" | "both"
    #
    #     # ========== 策略参数 ==========
    #     entry_period=20,
    #     exit_period=10,
    #     n_value=2.0,
    #     risk_percent=1.0,
    #     initial_capital=200000.0,
    #
    #     # ========== 加仓（仅 long_more） ==========
    #     # max_pyramid=5,
    # )
    # ================ 泡泡玛特 ================
    config = BacktestConfig(
        ts_code="09992.HK",
        stock_name="泡泡玛特",
        market="HK",
        mode="long_more",
        years=5,
        per_shares=200,
        dividend_info=pd.DataFrame({
            "ex_date": ["20260515"],
            "dividend": [2.7248],
        }),
        max_pyramid=5,
        skip_first_add = True
    )
    run(config)
