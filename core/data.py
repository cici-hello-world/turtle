"""数据获取与预处理 —— A股 / 港股统一入口"""

import os
import pandas as pd
import tushare as ts
from exchange_calendars import get_calendar


def fetch_data(ts_code: str, stock_name: str, market: str,
               start_date: str, end_date: str, exchange: str,
               dividend_info: pd.DataFrame | None = None) -> pd.DataFrame:
    """获取并预处理股票数据，返回标准格式 DataFrame

    返回列: Date, Open, High, Low, Close, Volume, Amount, Adj_Factor
    """
    token = os.environ.get("TUSHARE_TOKEN")
    if not token:
        raise RuntimeError("❌ 错误：未设置 TUSHARE_TOKEN 环境变量")

    ts.set_token(token)
    pro = ts.pro_api()

    print(f"🚀 开始获取 {stock_name} 数据...")

    if market == "A":
        df = _fetch_a_share(pro, ts_code, start_date, end_date)
    else:
        df = _fetch_hk_share(pro, ts_code, start_date, end_date, dividend_info)

    # 检查数据完整性
    _check_data_completeness(df, exchange, start_date, end_date)

    print(f"✅ 数据准备完成：共 {len(df)} 条记录")
    return df


def _fetch_a_share(pro, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """A股数据获取：日线 + 复权因子 → 前复权"""
    # 1. 下载非复权日线
    df_daily = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
    if df_daily.empty:
        raise ValueError("❌ 错误：未获取到日线数据")

    # 2. 下载复权因子
    try:
        df_adj = pro.adj_factor(ts_code=ts_code, start_date=start_date, end_date=end_date)
    except Exception as e:
        print(f"⚠️ 警告：复权因子获取失败，尝试使用默认因子 1.0 ({e})")
        df_adj = pd.DataFrame({"trade_date": df_daily["trade_date"], "adj_factor": 1.0})

    # 3. 合并与计算前复权
    df = pd.merge(df_daily, df_adj, on="trade_date", how="inner")
    df = df.sort_values("trade_date")

    return _compute_qfq(df)


def _fetch_hk_share(pro, ts_code: str, start_date: str, end_date: str,
                    dividend_info: pd.DataFrame | None = None) -> pd.DataFrame:
    """港股数据获取：hk_daily_adj → 可选分红除权修正 → 前复权"""
    # 1. 下载复权日线
    df = pro.hk_daily_adj(ts_code=ts_code, start_date=start_date, end_date=end_date)
    if df.empty:
        raise ValueError("❌ 错误：未获取到日线数据")

    df = df.sort_values("trade_date")

    # 2. 手动修正分红除权因子
    if dividend_info is not None:
        modify_adj_factor(df, dividend_info)

    return _compute_qfq(df)


def _compute_qfq(df: pd.DataFrame) -> pd.DataFrame:
    """前复权计算 + 列重命名为标准格式"""
    latest_factor = df["adj_factor"].iloc[-1]
    df["Open_QFQ"] = df["open"] * (df["adj_factor"] / latest_factor)
    df["High_QFQ"] = df["high"] * (df["adj_factor"] / latest_factor)
    df["Low_QFQ"] = df["low"] * (df["adj_factor"] / latest_factor)
    df["Close_QFQ"] = df["close"] * (df["adj_factor"] / latest_factor)

    df = df.rename(columns={"trade_date": "Date"})
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[["Date", "Open_QFQ", "High_QFQ", "Low_QFQ", "Close_QFQ", "vol", "amount", "adj_factor"]]
    df.columns = ["Date", "Open", "High", "Low", "Close", "Volume", "Amount", "Adj_Factor"]
    return df


def modify_adj_factor(df: pd.DataFrame, dividend_info: pd.DataFrame):
    """手动修正港股复权因子（处理 tushare 未包含的分红除权事件）"""
    print(f"🚀 开始处理 {len(dividend_info)} 个除权事件...\n")

    for idx, row in dividend_info.iterrows():
        ex_date = row["ex_date"]
        dividend_amount = row["dividend"]

        print(f"🔍 处理除权日: {ex_date}, 分红: {dividend_amount} HKD")

        # 检查该日期是否在股价数据中
        if ex_date not in df["trade_date"].values:
            print(f"⚠️  警告：股价数据中未找到日期 {ex_date}，跳过。")
            continue

        # 获取该除权日在股价数据中的索引
        ex_idx = df[df["trade_date"] == ex_date].index[0]

        # 获取前一天收盘价
        mask_prev = df["trade_date"] < ex_date
        prev_row = df[mask_prev].iloc[-1]
        prev_close = prev_row["close"]

        if prev_close > dividend_amount:
            ex_price = prev_close - dividend_amount
            ex_ratio = ex_price / prev_close
            print(f"✅ 除权计算:")
            print(f"   除权前收盘价: {prev_close:.2f} HKD")
            print(f"   分红金额: {dividend_amount:.4f} HKD")
            print(f"   除权参考价: {ex_price:.2f} HKD")
            print(f"   调整系数 (Ex-Ratio): {ex_ratio:.6f}")

            # 修正除权日之后的因子
            mask_future = df.index > ex_idx
            df.loc[mask_future, "adj_factor"] = df.loc[mask_future, "adj_factor"] * ex_ratio
            print(f"✅ 已修正 {mask_future.sum()} 条数据的复权因子 (乘以 {ex_ratio:.6f})")
        else:
            print(f"❌ 错误：除权前收盘价 ({prev_close}) 小于分红 ({dividend_amount})，数据异常！")


def _check_data_completeness(df: pd.DataFrame, exchange: str,
                              start_date: str, end_date: str):
    """检查数据是否缺失交易日"""
    calendar = get_calendar(exchange)
    trading_days = calendar.sessions_in_range(start_date, end_date)
    trading_days_no_tz = trading_days.tz_localize(None)

    label = "A股" if exchange in ("SSE", "XSHG") else "港股"
    print(f"{label}交易日数：{len(trading_days)}")

    if len(trading_days) != len(df):
        missing_dates = trading_days_no_tz[~trading_days_no_tz.isin(df["Date"])]
        print(f"✅ 数据有误：缺失共 {len(trading_days) - len(df)} 条记录")
        print("缺失的日期如下：")
        print(missing_dates.strftime("%Y-%m-%d").tolist())
