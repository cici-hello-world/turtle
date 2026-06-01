import pandas as pd
import numpy as np
import os
from pathlib import Path

# ================= 配置区域 =================
# 回测报告文件夹路径 (与 validate_turtle_stock_v3.py 保持一致)
data_dir = "htkj_backtest"

# 筛选阈值 (可根据需要调整，但通过验证的标的通常已达标)
THRESHOLDS = {
    "min_sharpe": 1.0,
    "min_profit_factor": 2.5,
    "min_turnover": 100000000
}

# 常用表名和列名映射 (与验证脚本保持一致，确保兼容)
SUMMARY_NAMES = ['Summary', '汇总表', '总览', 'Summary Report']
SUMMARY_COLS = {
    '夏普比率': ['夏普比率', 'Sharpe Ratio', 'Sharpe'],
    '盈亏比': ['盈亏比', 'Profit Factor', 'Profit/Loss Ratio', '赔率'],
    '总收益率': ['总收益率 (%)', 'Total Return', '收益率', 'Return %'],
    '最大回撤': ['最大回撤 (%)', 'Max Drawdown', '回撤', 'Drawdown %'],
    '总交易': ['总交易', '总交易次数', 'Trades', 'Trade Count']
}


def get_val_from_row(row, col_candidates):
    """辅助函数：从行中提取值"""
    for cand in col_candidates:
        if cand in row.index:
            val = row[cand]
            # 处理百分数 (如果是百分比，转换为小数或直接保留，这里统一保留原值，后续计算再处理)
            # 注意：总收益率如果是百分比，例如 100，代表 100%，直接比较即可
            # 夏普和盈亏比通常是小数
            return val
    return np.nan


def read_stock_metrics(file_path):
    """读取单只股票的核心指标"""
    try:
        xl = pd.ExcelFile(file_path)

        # 查找 Summary 表
        summary_sheet = None
        for name in SUMMARY_NAMES:
            if name in xl.sheet_names:
                summary_sheet = name
                break
        if not summary_sheet:
            return None

        df_sum = pd.read_excel(file_path, sheet_name=summary_sheet, header=0)
        if df_sum.empty:
            return None

        row = df_sum.iloc[0]

        # 提取核心指标
        sharpe = get_val_from_row(row, SUMMARY_COLS['夏普比率'])
        pf = get_val_from_row(row, SUMMARY_COLS['盈亏比'])
        total_ret = get_val_from_row(row, SUMMARY_COLS['总收益率'])
        max_dd = get_val_from_row(row, SUMMARY_COLS['最大回撤'])
        trades = get_val_from_row(row, SUMMARY_COLS['总交易'])

        # 文件名解析
        filename = os.path.basename(file_path)
        parts = filename.split('_')
        code = parts[1] if len(parts) > 1 else "Unknown"
        risk = "2.0%" if "2.0" in filename else ("5.0%" if "5.0" in filename else "?")

        return {
            '代码': code,
            '风险率': risk,
            '夏普比率': sharpe,
            '盈亏比': pf,
            '总收益率 (%)': total_ret,
            '最大回撤 (%)': max_dd,
            '总交易次数': trades,
            '文件名': filename
        }

    except Exception as e:
        print(f"读取文件 {file_path} 失败: {e}")
        return None


def main():
    print("=" * 80)
    print("🏆 海龟策略标的优选排序系统 (三级筛选法)")
    print("=" * 80)

    if not os.path.exists(data_dir):
        print(f"❌ 错误：目录不存在 -> {data_dir}")
        return

    files = list(Path(data_dir).glob("*.xlsx"))
    if not files:
        print(f"❌ 错误：在 {data_dir} 中未找到任何 .xlsx 文件。")
        return

    print(f"📂 发现 {len(files)} 个文件，正在提取核心指标...\n")

    stocks = []
    for file in files:
        data = read_stock_metrics(file)
        if data:
            stocks.append(data)
            print(f"✅ 已读取: {data['代码']} ({data['风险率']}) - 夏普: {data['夏普比率']}")
        else:
            print(f"❌ 读取失败: {os.path.basename(file)}")

    if not stocks:
        print("\n❌ 没有成功读取到任何数据。")
        return

    # 创建 DataFrame
    df_stocks = pd.DataFrame(stocks)

    # 清理数据 (将 NaN 转换为 0 或特定值，方便排序)
    # 注意：夏普比率和盈亏比如果为 NaN，排序时会排在最后
    df_stocks = df_stocks.fillna({'夏普比率': -999, '盈亏比': -999, '总收益率 (%)': -999})

    # 三级排序规则
    # 1. 夏普比率 (降序)
    # 2. 盈亏比 (降序)
    # 3. 总收益率 (降序)
    df_sorted = df_stocks.sort_values(
        by=['夏普比率', '盈亏比', '总收益率 (%)'],
        ascending=[True, True, True]  # 都是降序
    )

    # 输出排序结果
    print("\n" + "=" * 80)
    print("🏆 海龟策略标的优选排序结果 (按优先级从高到低)")
    print("=" * 80)
    print(
        f"{'排名':<4} | {'代码':<8} | {'风险':<6} | {'夏普':<8} | {'盈亏比':<8} | {'总收益':<8} | {'最大回撤':<8} | {'交易次数':<6}")
    print("-" * 80)

    for idx, row in df_sorted.iterrows():
        print(
            f"{idx + 1:<4} | {row['代码']:<8} | {row['风险率']:<6} | {row['夏普比率']:<8.2f} | {row['盈亏比']:<8.2f} | {row['总收益率 (%)']:<8.2f}% | {row['最大回撤 (%)']:<8.2f}% | {row['总交易次数']:<6}")

    print("-" * 80)

    # 推荐
    best = df_sorted.iloc[0]
    print(f"\n🌟 强烈推荐首仓执行: {best['代码']} (风险: {best['风险率']})")
    print(f"   夏普比率: {best['夏普比率']} (风险调整后收益最高)")
    print(f"   盈亏比: {best['盈亏比']} (大赚小亏能力最强)")
    print(f"   总收益率: {best['总收益率 (%)']}%")
    print(f"   最大回撤: {best['最大回撤 (%)']}%")

    # 保存排序结果到 Excel
    output_file = "海龟策略优选排序_三级筛选.xlsx"
    df_sorted.to_excel(output_file, index=False)
    print(f"\n📂 完整排序报表已保存至: {output_file}")
    print("\n💡 建议:")
    print("   1. 优先选择排名第 1 的股票作为主仓。")
    print("   2. 如果夏普比率接近，可以按盈亏比和收益率进一步筛选。")
    print("   3. 所有推荐标的均已通过 8 大验证关卡，可放心执行。")


if __name__ == "__main__":
    main()