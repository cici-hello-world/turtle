"""报告生成 —— 控制台打印 + Excel 输出"""

import os
import pandas as pd


def print_summary(stats: dict, config):
    """控制台打印回测结果摘要"""
    print("\n" + "=" * 70)
    print("📊 海龟策略回测结果 (带动态仓位管理)")
    print("=" * 70)

    print(f"   交易次数：{stats['总交易']}")
    print(f"   胜率：{stats['胜率 (%)']}%")
    print(f"   赔率：{stats['赔率']}")
    print(f"   总盈亏：{stats['总盈亏 (¥)']:,.2f}¥")
    print(f"   总收益率：{stats['总收益率 (%)']}%")
    print(f"   超额收益：{stats['超额收益 (%)']}%")
    print(f"   最大回撤：{stats['最大回撤 (%)']}%")
    print(f"   最终资金：{stats['最终资金 (¥)']:,.2f}¥")
    print(f"   夏普比率：{stats['夏普比率']}")
    print(f"   CAGR (%)：{stats['CAGR (%)']}")
    print(f"   MAR比率：{stats['MAR比率']}")
    print(f"   衰落持续期 (天)：{stats['衰落持续期 (天)']}")
    print(f"   回归年度回报率 (%)：{stats['回归年度回报率 (%)']}")
    print(f"   R平方：{stats['R平方']}")
    print(f"   R立方：{stats['R立方']}")
    print(f"   稳健夏普比率：{stats['稳健夏普比率']}")

    print("=" * 70)


def save_excel_report(df: pd.DataFrame, trades: list[dict], stats: dict, config):
    """保存 Excel 报告到 output_dir"""
    os.makedirs(config.output_dir, exist_ok=True)

    # 构造文件名
    prefix = "Backtest_Report"
    if config.mode == "long_more":
        prefix += "_more"
    if config.mode == "anti_martingale":
        prefix += "_AM"
    if config.mode == "position_management_1":
        prefix += "_PM"
    report_path = os.path.join(
        config.output_dir,
        f"{prefix}_{config.channel_type}_{config.ts_code}_{config.start_date}.xlsx",
    )

    with pd.ExcelWriter(report_path, engine="openpyxl") as writer:
        # Summary 表
        summary_data = [
            ["单笔风险", f"{config.risk_percent}%"],
            ["回测周期", f"{config.years}年"],
            ["交易次数", stats["总交易"]],
            ["胜率 (%)", stats["胜率 (%)"]],
            ["赔率", stats["赔率"]],
            ["期望收益率", stats["期望收益率"]],
            ["总盈亏 (¥)", stats["总盈亏 (¥)"]],
            ["总收益率 (%)", stats["总收益率 (%)"]],
            ["超额收益 (%)", stats["超额收益 (%)"]],
            ["最大回撤 (%)", stats["最大回撤 (%)"]],
            ["最终资金 (¥)", stats["最终资金 (¥)"]],
            ["夏普比率", stats["夏普比率"]],
            ["CAGR (%)", stats["CAGR (%)"]],
            ["MAR比率", stats["MAR比率"]],
            ["衰落持续期 (天)", stats["衰落持续期 (天)"]],
            ["回归年度回报率 (%)", stats["回归年度回报率 (%)"]],
            ["R平方", stats["R平方"]],
            ["R立方", stats["R立方"]],
            ["稳健夏普比率", stats["稳健夏普比率"]],
        ]
        pd.DataFrame(summary_data, columns=["指标", "数值"]).to_excel(
            writer, sheet_name="Summary", index=False
        )

        # 日线原始数据
        df_to_export = df.copy()
        df_to_export["Date"] = df_to_export["Date"].dt.strftime("%Y-%m-%d")
        df_to_export.to_excel(writer, sheet_name="日线原始数据", index=False)

        # 仓位管理策略 — 独立 sheet
        if config.mode == "position_management_1":
            if trades:
                df_pm = pd.DataFrame(trades)
                pm_cols = ["Date", "Type", "Price", "Shares", "Amount",
                           "Position_After", "Capital_After", "Equity_After",
                           "Reference_Price", "Reason"]
                df_pm[pm_cols].to_excel(writer, sheet_name="仓位管理明细", index=False)
        else:
            # 标准策略交易明细
            if trades:
                df_trades = pd.DataFrame(trades)
                df_trades["Entry Date"] = pd.to_datetime(df_trades["Entry Date"]).dt.strftime("%Y-%m-%d")
                df_trades["Exit Date"] = pd.to_datetime(df_trades["Exit Date"]).dt.strftime("%Y-%m-%d")
                df_trades.to_excel(writer, sheet_name="System 1 交易明细", index=False)

    print(f"\n✅ 报告已保存：{os.path.abspath(report_path)}")
    print(f"\n📁 Excel 文件包含以下工作表:")
    print(f"   1️⃣  Summary - 策略汇总统计 (含资金曲线指标)")
    print(f"   2️⃣  日线原始数据 - {len(df)} 条日线数据")
    if trades:
        if config.mode == "position_management_1":
            print(f"   3️⃣  仓位管理明细 - {len(trades)} 笔加减仓交易")
        else:
            print(f"   3️⃣   System 1 交易明细 - {len(trades)} 笔交易 (含股数/盈亏金额)")
