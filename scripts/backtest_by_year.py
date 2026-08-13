#!/usr/bin/env python3
"""
按指定月日分段，每年独立回测，汇总到一个 Excel。

每年区间:
    pm_start_date = max(原始pm_start_date, year-mm-dd)
    end_date      = (year+1)-mm-dd 的前一天

结果直接取自 run_backtest.py 输出的 Excel 文件。

用法:
    python scripts/backtest_by_year.py --ts-code 01810.HK -m 08 -d 06
    python scripts/backtest_by_year.py --ts-code 06181.HK -m 07 -d 29 -s 2024 -e 2025
    python scripts/backtest_by_year.py --ts-code 02015.HK -m 09 -d 09     # bb_rebound 标的自动识别
    python scripts/backtest_by_year.py --ts-code 02015.HK -m 09 -d 09 --mode position_management_1
"""

import argparse
import os
import sys
from datetime import datetime, timedelta

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.config import BacktestConfig
from backtest_entry.run_backtest import run
from strategy.position_management_1.common_config import STOCKS, OPTIMAL_PARAMS
from strategy.bb_rebound.bb_config import BB_STOCKS


def main():
    parser = argparse.ArgumentParser(description="按年度分段回测并汇总")
    parser.add_argument("--ts-code", type=str, required=True, help="股票代码")
    parser.add_argument("-m", "--month", type=int, required=True, help="起始月份")
    parser.add_argument("-d", "--day", type=int, required=True, help="起始日期")
    parser.add_argument("-s", "--start-year", type=int, default=None, help="起始年份")
    parser.add_argument("-e", "--end-year", type=int, default=None, help="结束年份")
    parser.add_argument("-o", "--output", type=str, default=None, help="输出路径")
    parser.add_argument("--mode", type=str, default=None,
                        help="策略模式 (默认自动: bb_rebound 配置优先, 否则 position_management_1)")
    args = parser.parse_args()

    ts_code = args.ts_code
    month, day = args.month, args.day

    # 解析模式与配置源
    if args.mode is None:
        mode = "bb_rebound" if ts_code in BB_STOCKS else "position_management_1"
    else:
        mode = args.mode
    if mode == "bb_rebound":
        if ts_code not in BB_STOCKS:
            print(f"❌ 未配置 {ts_code} 的 bb_rebound 参数，请先在 strategy/bb_rebound/bb_config.py 中添加")
            sys.exit(1)
        stock_cfg = dict(BB_STOCKS[ts_code])
        optimal = {}
    else:
        if ts_code not in STOCKS:
            print(f"❌ 未找到 {ts_code} 的配置"); sys.exit(1)
        stock_cfg = dict(STOCKS[ts_code])
        optimal = OPTIMAL_PARAMS.get(ts_code, {})

    pm_start = datetime.strptime(stock_cfg["pm_start_date"], "%Y%m%d")
    start_year = args.start_year or pm_start.year
    end_year = args.end_year or datetime.today().year

    # 输出目录与文件名规则跟随 core/reporter.py
    probe = BacktestConfig(**{**stock_cfg, **optimal})
    output_dir = probe.output_dir
    os.makedirs(output_dir, exist_ok=True)
    prefix = "Backtest_Report" + ("_PM" if mode == "position_management_1" else "")
    raw_name = f"{prefix}_{probe.channel_type}_{ts_code}_{probe.start_date}.xlsx"
    trade_sheet = "仓位管理明细" if mode == "position_management_1" else "System 1 交易明细"
    yearly_suffix = "PM_donchian" if mode == "position_management_1" else mode
    out_path = args.output or os.path.join(
        output_dir, f"Backtest_Report_{yearly_suffix}_{ts_code}_yearly.xlsx")

    summaries = []
    trade_sheets = {}
    temp_files = []

    for year in range(start_year, end_year):
        y_start = datetime(year, month, day)
        actual_start = max(pm_start, y_start)
        y_end = datetime(year + 1, month, day) - timedelta(days=1)

        if actual_start >= y_end:
            continue

        s_str = actual_start.strftime("%Y%m%d")
        e_str = y_end.strftime("%Y%m%d")
        label = str(year)

        print(f"📊 {year}: {s_str} → {e_str} ...", end=" ", flush=True)

        # 构建 config，覆盖 pm_start_date 和 end_date
        kw = {**stock_cfg, **optimal, "pm_start_date": s_str, "end_date": e_str}
        run(BacktestConfig(**kw))

        # reporter 用 start_date 命名，每年会覆盖；重命名为带年份的临时文件
        raw_file = os.path.join(output_dir, raw_name)
        report_file = os.path.join(
            output_dir, f"{prefix}_{probe.channel_type}_{ts_code}_{label}.xlsx")
        if os.path.exists(raw_file):
            os.rename(raw_file, report_file)
        temp_files.append(report_file)

        if not os.path.exists(report_file):
            print("❌ 文件未生成")
            continue

        # 读取 Summary 和 仓位管理明细
        try:
            summary_df = pd.read_excel(report_file, sheet_name="Summary")
            trades_df = pd.read_excel(report_file, sheet_name=trade_sheet)
        except Exception as ex:
            print(f"❌ 读取失败: {ex}")
            continue

        # 提取关键指标
        sdict = dict(zip(summary_df["指标"], summary_df["数值"]))
        summaries.append({
            "年份": label, "起始日": s_str, "结束日": e_str,
            "总收益率": sdict.get("总收益率 (%)", "N/A"),
            "超额收益": sdict.get("超额收益 (%)", "N/A"),
            "最大回撤": sdict.get("最大回撤 (%)", "N/A"),
            "夏普比率": sdict.get("夏普比率", "N/A"),
            "胜率": sdict.get("胜率 (%)", "N/A"),
            "交易次数": sdict.get("交易次数", "N/A"),
            "最终权益": sdict.get("最终资金 (¥)", "N/A"),
        })
        trade_sheets[label] = trades_df
        print("✅")

    # 合并写入
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        pd.DataFrame(summaries).to_excel(writer, sheet_name="年度汇总", index=False)
        for label, tdf in trade_sheets.items():
            sheet_name = f"{label}交易明细"[:31]  # Excel sheet name max 31 chars
            tdf.to_excel(writer, sheet_name=sheet_name, index=False)

    # 删除临时文件
    for f in temp_files:
        if os.path.exists(f):
            os.remove(f)

    print(f"\n✅ {out_path}")
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
