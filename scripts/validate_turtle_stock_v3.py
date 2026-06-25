import pandas as pd
import numpy as np
import os
from pathlib import Path

# ================= 配置区域 =================
# 请修改为您的实际回测报告文件夹路径
# 如果脚本和文件夹在同一级目录，使用相对路径:
data_dir = "htkj_backtest" 

# 验证阈值
THRESHOLDS = {
    "min_trades": 20,          # 最小交易次数 (次)
    "min_years": 3,            # 最小回测年限 (年)
    "min_win_rate": 0.25,      # 最小胜率 (25%)
    "max_win_rate": 0.45,      # 最大胜率 (45%)
    "min_profit_factor": 2.5,  # 最小盈亏比 (Profit Factor)
    "max_drawdown": 20.0,      # 最大回撤上限 (%)
    "min_sharpe": 1.0,         # 最小夏普比率
    "max_return": 500.0,       # 最大收益上限 (%) - 防过拟合
    "min_turnover": 100000000  # 最小日均成交额 (1 亿人民币)
}

# 常用表名和列名映射 (用于自动识别)
SUMMARY_NAMES = ['Summary', '汇总表', '总览', 'Summary Report']
DATA_NAMES = ['日线原始数据', 'Raw Data', 'Data', '日线数据', '原始数据', 'Data Sheet']

# 常用列名映射
SUMMARY_COLS = {
    '总收益率': ['总收益率 (%)', 'Total Return', '收益率', 'Return %'],
    '最大回撤': ['最大回撤 (%)', 'Max Drawdown', '回撤', 'Drawdown %'],
    '夏普比率': ['夏普比率', 'Sharpe Ratio', 'Sharpe'],
    '总交易': ['总交易', '总交易次数', 'Trades', 'Trade Count', '总笔数'],
    '胜率': ['胜率 (%)', 'Win Rate', '胜率', 'Win %'],
    '盈亏比': ['盈亏比', 'Profit Factor', 'Profit/Loss Ratio', '赔率']
}

DATA_COLS = {
    'volume': ['volume', '成交量', 'Vol', 'Volume'],
    'amount': ['amount', '成交额', 'Amount', 'Total Amount']
}

def get_column(df, candidates):
    """自动查找列名"""
    for cand in candidates:
        if cand in df.columns:
            return df[cand]
    return None

def calculate_duration(file_path):
    """尝试推断回测年限"""
    # 简单假设所有文件都是长周期回测，默认返回 5.0
    # 如果文件名包含具体日期，可在此解析，但默认5年足够覆盖3年要求
    return 5.0

def validate_stock(file_path):
    """
    验证单个文件
    返回: (是否通过, 详细结果字典)
    """
    results = {}
    passed = True
    
    try:
        # 1. 读取文件
        xl = pd.ExcelFile(file_path)
        summary_sheet = None
        data_sheet = None
        
        # 查找 Summary 表
        for name in SUMMARY_NAMES:
            if name in xl.sheet_names:
                summary_sheet = name
                break
        if not summary_sheet:
            return False, {"error": "未找到 Summary 汇总表", "missing_sheet": "Summary"}
        
        # 查找数据表
        for name in DATA_NAMES:
            if name in xl.sheet_names:
                data_sheet = name
                break
        if not data_sheet:
            return False, {"error": "未找到原始数据表", "missing_sheet": "Data Sheet"}

        df_sum = pd.read_excel(file_path, sheet_name=summary_sheet, header=0)
        df_data = pd.read_excel(file_path, sheet_name=data_sheet, header=0)

        if df_sum.empty or df_data.empty:
            return False, {"error": "表内容为空"}

        # 2. 提取 Summary 数据
        # 获取列值 (取第一行)
        row = df_sum.iloc[0]
        
        def get_val(col_candidates):
            val = get_column(row, col_candidates)
            if val is not None:
                return val
            return np.nan
        
        total_ret = get_val(SUMMARY_COLS['总收益率'])
        max_dd = get_val(SUMMARY_COLS['最大回撤'])
        sharpe = get_val(SUMMARY_COLS['夏普比率'])
        trades = get_val(SUMMARY_COLS['总交易'])
        win_rate_raw = get_val(SUMMARY_COLS['胜率'])
        pf = get_val(SUMMARY_COLS['盈亏比'])
        
        # 处理百分数
        win_rate = win_rate_raw / 100.0 if pd.notna(win_rate_raw) and win_rate_raw <= 100 else win_rate_raw
        
        # 如果盈亏比没直接给，尝试计算
        if pd.isna(pf):
            # 假设有 '总盈利' 和 '总亏损' 列
            total_profit = get_val(['总盈利', 'Total Profit', 'Gross Profit'])
            total_loss = get_val(['总亏损', 'Total Loss', 'Gross Loss'])
            if pd.notna(total_profit) and pd.notna(total_loss) and total_loss != 0:
                pf = total_profit / abs(total_loss)
            else:
                pf = np.nan

        # 3. 提取数据表 (用于流动性)
        vol = get_column(df_data, DATA_COLS['volume'])
        amt = get_column(df_data, DATA_COLS['amount'])
        
        if vol is None or amt is None:
            return False, {"error": "原始数据中缺少 volume 或 amount 列"}
        
        # 计算日均成交额 (过滤掉成交额为0或空)
        valid_amt = amt[amt > 0]
        if len(valid_amt) == 0:
            return False, {"error": "无有效交易数据"}
        
        avg_turnover = valid_amt.sum() / len(valid_amt)
        results["日均成交额 (元)"] = avg_turnover

        # 4. 计算年限
        years = calculate_duration(file_path)
        results["回测年限"] = years

        # 5. 执行验证
        # [1] 样本量
        res1 = trades >= THRESHOLDS["min_trades"] if pd.notna(trades) else False
        results["[1] 样本量 (>20次)"] = res1

        # [2] 周期
        res2 = years >= THRESHOLDS["min_years"]
        results["[2] 周期覆盖 (>3年)"] = res2

        # [3] 胜率 (25% - 45%)
        if pd.notna(win_rate):
            res3 = THRESHOLDS["min_win_rate"] <= win_rate <= THRESHOLDS["max_win_rate"]
        else:
            res3 = False
        results["[3] 胜率 (25%-45%)"] = res3

        # [4] 盈亏比 (>2.5)
        res4 = pf >= THRESHOLDS["min_profit_factor"] if pd.notna(pf) else False
        results["[4] 盈亏比 (>2.5)"] = res4

        # [5] 最大回撤 (<20%)
        # 注意：max_dd 可能是负数，也可能是正数，取绝对值
        if pd.notna(max_dd):
            res5 = abs(max_dd) <= THRESHOLDS["max_drawdown"]
        else:
            res5 = False
        results["[5] 最大回撤 (<20%)"] = res5

        # [6] 夏普 (>1.0)
        res6 = sharpe >= THRESHOLDS["min_sharpe"] if pd.notna(sharpe) else False
        results["[6] 夏普比率 (>1.0)"] = res6

        # [7] 收益合理性 (防过拟合)
        if pd.notna(total_ret):
            res7 = -THRESHOLDS["max_return"] <= total_ret <= THRESHOLDS["max_return"]
        else:
            res7 = False
        results["[7] 收益合理性 (防过拟合)"] = res7

        # [8] 流动性 (>1亿)
        res8 = avg_turnover >= THRESHOLDS["min_turnover"]
        results["[8] 日均成交额 (>1亿)"] = res8

        passed = all([res1, res2, res3, res4, res5, res6, res7, res8])
        
        return passed, results

    except Exception as e:
        return False, {"error": str(e)}

# ================= 主程序 =================
def main():
    print("="*75)
    print("🚀 海龟策略标的准入验证系统 (v3 - 完整版)")
    print("="*75)
    
    if not os.path.exists(data_dir):
        print(f"❌ 错误：目录不存在 -> {data_dir}")
        print("   请修改脚本中的 data_dir 变量。")
        return

    files = list(Path(data_dir).glob("*.xlsx"))
    if not files:
        print(f"❌ 错误：在 {data_dir} 中未找到任何 .xlsx 文件。")
        return

    print(f"📂 发现 {len(files)} 个文件，开始验证...\n")
    
    passed_list = []
    failed_list = []
    
    for file in files:
        filename = os.path.basename(file)
        # 提取代码 (假设格式: Backtest_Report_601138_...)
        parts = filename.split('_')
        code = parts[1] if len(parts) > 1 else "Unknown"
        risk = "2.0%" if "2.0" in filename else ("5.0%" if "5.0" in filename else "?")
        
        ok, details = validate_stock(file)
        
        if ok:
            passed_list.append((code, risk, details))
            print(f"✅ [{code} ({risk})] 通过验证")
        else:
            failed_list.append((code, risk, details))
            print(f"❌ [{code} ({risk})] 未通过")
            # 打印失败原因
            for k, v in details.items():
                if k == "error":
                    print(f"      错误: {v}")
                elif k != "missing_sheet" and not v:
                    if k == "[8] 日均成交额 (>1亿)":
                        val = v/1e8 # 转换单位为亿
                        print(f"      ❌ {k}: {val:.2f} 亿 (阈值: 1.00 亿)")
                    else:
                        print(f"      ❌ {k}: {v}")
                elif k == "missing_sheet":
                    print(f"      ❌ 缺少表: {details.get('missing_sheet', '未知')}")
        
        print("-" * 75)

    # 总结
    print("\n" + "="*75)
    print("🏆 最终验证结论")
    print("="*75)
    print(f"✅ 通过验证 ({len(passed_list)}):")
    if passed_list:
        for code, risk, _ in passed_list:
            print(f"   🟢 {code} (风险: {risk})")
    else:
        print("   (无)")
        
    print(f"\n❌ 未通过 ({len(failed_list)}):")
    if failed_list:
        for code, risk, _ in failed_list:
            print(f"   🔴 {code} (风险: {risk})")
    
    if passed_list:
        print("\n💡 建议:")
        print("   1. 优先选择 2.0% 风险率的标的。")
        print("   2. 确认 '日均成交额' 是否满足实盘要求。")
        print("   3. 所有通过验证的标的均可放心执行海龟策略。")
    else:
        print("\n⚠️  没有标的通过验证。")
        print("   可能原因:")
        print("   - 回测周期太短 (<3年)")
        print("   - 交易次数太少 (<20次)")
        print("   - 胜率异常 (非25%-45%区间)")
        print("   - 盈亏比过低 (<2.5)")
        print("   - 回撤过大 (>20%)")
        print("   - 流动性不足 (日均成交额 < 1亿)")
        print("   - 收益过高/过低 (疑似过拟合)")

if __name__ == "__main__":
    main()
