import tushare as ts
import pandas as pd
import numpy as np
import os
from exchange_calendars import get_calendar


# ================= 配置区域 =================
print(f"🚀 ================= 版本1================= ")

STOCK_CODE = "601138.SH"
STOCK_NAME = "工业富联"
# STOCK_CODE = "000625.SZ"
# STOCK_NAME = "长安汽车"
TODAY = pd.Timestamp.today()
YEARS = 8
START_DATE = (TODAY - pd.DateOffset(years=YEARS)).strftime('%Y%m%d')
END_DATE = TODAY.strftime('%Y%m%d')
EXCHANGE = "SSE"        # A股上交所SSE / 深交所SZSE / 港股HKEX

OUTPUT_DIR = "a_stock_backtest_short"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 海龟策略参数
N_VALUE = 2.0  # 止损倍数 (海龟标准)
RISK_PERCENT = 1.0  # 单笔风险 1% (用于计算仓位)
INITIAL_CAPITAL = 200000.0  # 初始资金 10 万元
# ================= 1. 数据获取与处理 =================

def fetch_and_preprocess_data():
    token = os.environ.get("TUSHARE_TOKEN")
    if not token:
        raise RuntimeError("❌ 错误：未设置 TUSHARE_TOKEN 环境变量")

    ts.set_token(token)
    pro = ts.pro_api()

    print(f"🚀 开始获取 {STOCK_NAME} 数据...")

    # 1. 下载非复权日线
    df_daily = pro.daily(ts_code=STOCK_CODE, start_date=START_DATE, end_date=END_DATE)
    if df_daily.empty:
        raise ValueError("❌ 错误：未获取到日线数据")

    # 2. 下载复权因子
    try:
        df_adj = pro.adj_factor(ts_code=STOCK_CODE, start_date=START_DATE, end_date=END_DATE)
    except Exception as e:
        print(f"⚠️ 警告：复权因子获取失败，尝试使用默认因子 1.0 ({e})")
        df_adj = pd.DataFrame({'trade_date': df_daily['trade_date'], 'adj_factor': 1.0})

    # 3. 合并与计算前复权
    df = pd.merge(df_daily, df_adj, on='trade_date', how='inner')
    df = df.sort_values('trade_date')

    # 前复权计算
    latest_factor = df['adj_factor'].iloc[-1]
    df['Open_QFQ'] = df['open'] * (df['adj_factor'] / latest_factor)
    df['High_QFQ'] = df['high'] * (df['adj_factor'] / latest_factor)
    df['Low_QFQ'] = df['low'] * (df['adj_factor'] / latest_factor)
    df['Close_QFQ'] = df['close'] * (df['adj_factor'] / latest_factor)

    # 重命名列
    df = df.rename(columns={'trade_date': 'Date'})
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[['Date', 'Open_QFQ', 'High_QFQ', 'Low_QFQ', 'Close_QFQ', 'vol', 'amount', 'adj_factor']]
    df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Amount', 'Adj_Factor']

    # 获取A股日历
    hk_calendar = get_calendar(EXCHANGE)

    # 计算交易日数
    trading_days = hk_calendar.sessions_in_range(START_DATE, END_DATE)
    trading_days_no_tz = trading_days.tz_localize(None)
    print(f"A股交易日数：{len(trading_days)}")
    if len(trading_days) != len(df):
        missing_dates = trading_days_no_tz[~trading_days_no_tz.isin(df['Date'])]
        print(f"✅ 数据有误：缺失共 {len(trading_days)-len(df)} 条记录")
        print("缺失的日期如下：")
        print(missing_dates.strftime('%Y-%m-%d').tolist())
    print(f"✅ 数据准备完成：共 {len(df)} 条记录")
    return df

# ================= 2. 海龟策略回测引擎 (带仓位管理) =================

def run_backtest_with_position_sizing(df, system_name, entry_period, exit_period, initial_capital=100000.0, risk_per_trade=0.01):
    print(f"\n🔍 开始运行 {system_name} (带动态仓位管理)...")
    print(f"   初始资金：{initial_capital:,.2f} | 单笔风险：{risk_per_trade*100:.1f}%")

    trades = []
    current_position = None
    entry_price = 0
    stop_loss_price = 0.0
    current_shares = 0
    entry_date = None

    # 资金管理变量
    capital = initial_capital
    equity_curve = [initial_capital]

    # 初始化指标
    df['TR'] = np.maximum(
        df['High'] - df['Low'],
        np.maximum(abs(df['High'] - df['Close'].shift(1)), abs(df['Low'] - df['Close'].shift(1)))
    )
    df['ATR'] = df['TR'].ewm(span=14, min_periods=14).mean()

    df['HH_Entry'] = df['High'].rolling(window=entry_period).max().shift(1)
    df['LL_Entry'] = df['Low'].rolling(window=entry_period).min().shift(1)
    df['HL_Exit'] = df['High'].rolling(window=exit_period).max().shift(1)
    df['LL_Exit'] = df['Low'].rolling(window=exit_period).min().shift(1)
    start_index = entry_period + 1

    for i in range(start_index, len(df)):
        row = df.iloc[i]
        current_date = row['Date']

        if current_position is None:
            #空仓
            if pd.isna(row['ATR']) or row['ATR'] == 0:
                continue

            atr = row['ATR']
            # 计算止损距离
            stop_dist = N_VALUE * atr
            target_risk_amount = capital * risk_per_trade
            # 入场shares手
            shares = int(target_risk_amount / (stop_dist * 100))
            if shares <= 0:
                continue
            entry_price = row['Close']
            # 做多信号
            if row['Close'] > row['HH_Entry']:
                # 检查资金是否足够 (保留 5% 缓冲)
                cost = row['Close'] * shares * 100
                if cost > capital * 0.95:
                    continue
                current_position = 'long'
                stop_loss_price = entry_price - stop_dist
                current_shares = shares
                entry_date = current_date
                capital -= cost
                print(f"  🟢 [{system_name}] 做多入场：{row['Close']:.2f} | 股数：{shares*100} | 投入：{cost:.2f} | 剩余资金：{capital:.2f} | 止损：{stop_loss_price:.2f}")
            elif row['Close'] < row['LL_Entry']:
                #做空信号
                income = row['Close'] * shares * 100
                current_position = 'short'
                stop_loss_price = entry_price + stop_dist
                current_shares = -shares
                entry_date = current_date
                capital += income
                print(f"  🟢 [{system_name}] 做空入场：{row['Close']:.2f} | 股数：{shares*100} | 收入：{income:.2f} | 剩余资金：{capital:.2f} | 止损：{stop_loss_price:.2f}")
        else:
            #持仓
            is_stopped = False
            exit_price = 0
            exit_reason = ""

            if current_position == 'long':#已测试
                # 优先检查硬止损/先实现方案一/方案二是当第一次出触发止损价则止损，需要拉取分时线数据
                if row['Close'] < stop_loss_price:
                    is_stopped = True
                    exit_price = row['Close']#考虑滑点
                    exit_reason = "Hard Stop Loss (2×ATR)"
                # 检查趋势反转
                elif row['Close'] < row['LL_Exit']:#已测试
                    is_stopped = True
                    exit_price = row['Close']#考虑滑点
                    exit_reason = "Trend Reversal (10d Low)"
            elif current_position == 'short':
                # 优先检查硬止损
                if row['Close'] > stop_loss_price:#已测试
                    is_stopped = True
                    exit_price = row['Close']#考虑滑点
                    exit_reason = "Hard Stop Loss (2×ATR)"
                # 检查趋势反转
                elif row['Close'] > row['HL_Exit']:#已测试
                    is_stopped = True
                    exit_price = row['Close'] #考虑滑点
                    exit_reason = "Trend Reversal (10d High)"

            if is_stopped:
                # 计算盈亏
                pnl_amount = (exit_price - entry_price) * current_shares * 100
                sell_proceeds = exit_price * current_shares * 100
                capital += sell_proceeds
                if current_position == 'long':
                    pnl_pct = (pnl_amount / (entry_price * current_shares * 100))
                elif current_position == 'short':
                    pnl_pct = (pnl_amount / abs(exit_price * current_shares * 100)) #可改为保证金
                trades.append({
                        'Entry Date': entry_date,
                        'Exit Date': current_date,
                        'Type': current_position,
                        'Entry Price': entry_price,
                        'Exit Price': exit_price,
                        'Shares': current_shares,
                        'Cost_¥': (entry_price * current_shares * 100) if current_position == 'long' else abs(exit_price * current_shares * 100),
                        'PnL_¥': pnl_amount,
                        'PnL_%': pnl_pct,
                        'Capital_After': capital,
                        'Reason': exit_reason
                    })

                equity_curve.append(capital)

                status = "🟢 盈利" if pnl_amount > 0 else "🔴 亏损"
                print(f"  {status} [{system_name}] 平仓：{exit_reason} | 盈亏：{pnl_amount:,.2f}¥ | 剩余资金：{capital:,.2f}")

                current_position = None
                current_shares = 0
                entry_price = 0
                stop_loss_price = 0.0
                entry_date = None

    # 强制平仓 (如果收盘时仍有持仓)
    if current_position is not None:
        last_price = df.iloc[-1]['Close']
        last_date = df.iloc[-1]['Date']
        pnl_amount = (last_price - entry_price) * current_shares * 100
        sell_proceeds = last_price * current_shares * 100
        if current_position == 'long':
            pnl_pct = (pnl_amount / (entry_price * current_shares * 100))
            capital += sell_proceeds
        elif current_position == 'short':
            pnl_pct = (pnl_amount / (exit_price * current_shares * 100))  # 可改为保证金
            capital -= sell_proceeds
        trades.append({
            'Entry Date': entry_date,
            'Exit Date': last_date,
            'Type': current_position,
            'Entry Price': entry_price,
            'Exit Price': last_price,
            'Shares': current_shares,
            'Cost_¥': (entry_price * current_shares * 100) if current_position == 'long' else (exit_price * current_shares * 100),
            'PnL_¥': pnl_amount,
            'PnL_%': pnl_pct,
            'Capital_After': capital,
            'Reason': "强制平仓"
        })
        # equity_curve.append(capital)
        print(f"  ⚠️ [{system_name}] 强制平仓：{pnl_amount:,.2f}¥ | 最终资金：{capital:,.2f}")
    return trades, equity_curve

# ================= 3. 统计计算 (基于资金曲线) =================

def calc_stats_with_equity(trades, initial_capital, equity_curve):
    if not trades:
        return {
            '总交易': 0, '胜率 (%)': 0, '赔率': 0, '期望收益率':0, '盈利次数': 0, '亏损次数': 0,
            '总盈亏 (¥)': 0, '总收益率 (%)': 0, '最大回撤 (%)': 0,
            '最终资金 (¥)': initial_capital, '夏普比率': 0, '平均盈亏 (¥)': 0
        }

    df_trades = pd.DataFrame(trades)
    df_trades = df_trades[df_trades['Reason'] != '强制平仓']
    wins = df_trades[df_trades['PnL_%'] > 0]
    losses = df_trades[df_trades['PnL_%'] < 0]
    final_capital = df_trades.iloc[-1]['Capital_After']
    print(f"最终资金:{final_capital}")
    total_trades = len(df_trades)
    win_count = len(wins)
    loss_count = len(losses)
    win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0

    total_pnl = final_capital - initial_capital
    total_return = (total_pnl / initial_capital) * 100

    # 最大回撤 (基于 equity_curve)
    equity_series = pd.Series(equity_curve)
    cummax = equity_series.cummax()
    drawdown = (equity_series - cummax) / cummax
    max_dd = drawdown.min() * 100

    # 平均盈亏
    avg_win = wins['PnL_¥'].mean() if win_count > 0 else 0
    avg_loss = abs(losses['PnL_¥'].mean()) if loss_count > 0 else 0
    win_loss_ratio = (avg_win / avg_loss) if avg_loss > 0 else 0

    # 期望值
    expectation = (win_rate/100 * win_loss_ratio) - (100-win_rate)/100
    #平均每次交易期望收益？
    expectation1 = (win_rate / 100 * avg_win) - ((100 - win_rate) / 100 * avg_loss)

    # 夏普比率 (简化版：假设无风险利率为0，年化收益率/年化波动率)
    # 这里计算日收益率的夏普比率 (简化为：平均日收益 / 日收益标准差 * sqrt(252))
    # 由于我们只有交易数据，这里用交易收益率来近似
    if len(equity_curve) > 1:
        returns = equity_series.pct_change().dropna()
        if len(returns) > 0 and returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            sharpe = 0
    else:
        sharpe = 0

    return {
        '总交易': total_trades,
        '胜率 (%)': round(win_rate, 2),
        '赔率': win_loss_ratio,
        '期望收益率': expectation,
        '盈利次数': win_count,
        '亏损次数': loss_count,
        '总盈亏 (¥)': round(total_pnl, 2),
        '总收益率 (%)': round(total_return, 2),
        '最大回撤 (%)': round(max_dd, 2),
        '最终资金 (¥)': round(final_capital, 2),
        '夏普比率': round(sharpe, 2),
        '平均盈亏 (¥)': round((avg_win + avg_loss)/2, 2)
    }

# ================= 4. 主程序与报告生成 =================

def main():
    try:
        # 1. 获取数据
        df = fetch_and_preprocess_data()

        # 2. 回测两个系统 (带仓位管理)
        trades_s1, equity_s1 = run_backtest_with_position_sizing(
            df, "System 1 (20/10+Stop)", 20, 10, INITIAL_CAPITAL, RISK_PERCENT/100
        )

        #trades_s2, final_cap_s2, equity_s2 = run_backtest_with_position_sizing(
        #    df, "System 2 (55/10+Stop)", 55, 10, INITIAL_CAPITAL, RISK_PERCENT/100
        #)

        # 3. 统计
        stats_s1 = calc_stats_with_equity(trades_s1, INITIAL_CAPITAL, equity_s1)

        # 4. 打印结果
        print("\n" + "="*70)
        print("📊 海龟策略回测结果 (带动态仓位管理)")
        print("="*70)

        print(f"   交易次数：{stats_s1['总交易']}")
        print(f"   胜率：{stats_s1['胜率 (%)']}%")
        print(f"   赔率：{stats_s1['赔率']}")
        print(f"   总盈亏：{stats_s1['总盈亏 (¥)']:,.2f}¥")
        print(f"   总收益率：{stats_s1['总收益率 (%)']}%")
        print(f"   最大回撤：{stats_s1['最大回撤 (%)']}%")
        print(f"   最终资金：{stats_s1['最终资金 (¥)']:,.2f}¥")
        print(f"   夏普比率：{stats_s1['夏普比率']}")


        print("="*70)

        # 5. 保存 Excel
        report_path = os.path.join(OUTPUT_DIR, f"Backtest_Report_{STOCK_CODE}_{START_DATE}起_{RISK_PERCENT}%单笔风险.xlsx")

        with pd.ExcelWriter(report_path, engine='openpyxl') as writer:
            # Summary 表
            summary_data = [
                ['单笔风险', f"{RISK_PERCENT}%"],
                ['回测周期', f"{YEARS}年"],
                ['交易次数', stats_s1['总交易']],
                ['胜率 (%)', stats_s1['胜率 (%)']],
                ['赔率', stats_s1['赔率']],
                ['期望收益率', stats_s1['期望收益率']],
                ['总盈亏 (¥)', stats_s1['总盈亏 (¥)']],
                ['总收益率 (%)', stats_s1['总收益率 (%)']],
                ['最大回撤 (%)', stats_s1['最大回撤 (%)']],
                ['最终资金 (¥)', stats_s1['最终资金 (¥)']],
                ['夏普比率', stats_s1['夏普比率']]
            ]
            pd.DataFrame(summary_data[1:], columns=summary_data[0]).to_excel(writer, sheet_name='Summary', index=False)

            # 日线原始数据
            df_to_export = df.copy()
            df_to_export['Date'] = df_to_export['Date'].dt.strftime('%Y-%m-%d')
            df_to_export.to_excel(writer, sheet_name='日线原始数据', index=False)

            # System 1 交易明细
            if trades_s1:
                df_trades_s1 = pd.DataFrame(trades_s1)
                df_trades_s1['Entry Date'] = pd.to_datetime(df_trades_s1['Entry Date']).dt.strftime('%Y-%m-%d')
                df_trades_s1['Exit Date'] = pd.to_datetime(df_trades_s1['Exit Date']).dt.strftime('%Y-%m-%d')
                df_trades_s1.to_excel(writer, sheet_name='System 1 交易明细', index=False)


        print(f"\n✅ 报告已保存：{os.path.abspath(report_path)}")
        print(f"\n📁 Excel 文件包含以下工作表:")
        print(f"   1️⃣  Summary - 策略汇总统计 (含资金曲线指标)")
        print(f"   2️⃣   日线原始数据 - {len(df)} 条日线数据")
        if trades_s1:
            print(f"   3️⃣   System 1 交易明细 - {len(trades_s1)} 笔交易 (含股数/盈亏金额)")

    except Exception as e:
        print(f"❌ 错误：{e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
