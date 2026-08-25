"""数据获取与预处理 —— A股 / 港股统一入口"""
import datetime
import os
import re
import pandas as pd
import tushare as ts
from exchange_calendars import get_calendar


def fetch_data(ts_code: str, stock_name: str, market: str,
               start_date: str, end_date: str, exchange: str,
               data_source: str = "tushare",
               use_cache: bool = True) -> pd.DataFrame:
    """获取并预处理股票数据，返回标准格式 DataFrame（不复权价格）

    返回列: Date, Open, High, Low, Close, Volume, Amount, Cash_Dividend

    data_source: "tushare" (默认) 或 "akshare" (东方财富数据源)
    use_cache:   True 时优先读本地缓存，命中则跳过网络请求
    """
    from core.cache import load as cache_load, save as cache_save, get_date_range

    # ---- 优先读缓存 ----
    if use_cache:
        cached = cache_load(ts_code, market)
        if cached is not None and not cached.empty:
            cache_start = cached["Date"].min().strftime("%Y%m%d")
            cache_end = cached["Date"].max().strftime("%Y%m%d")
            if cache_start <= start_date and cache_end >= end_date:
                result = cached[(cached["Date"] >= start_date) & (cached["Date"] <= end_date)].copy()
                print(f"📦 [缓存命中] {stock_name}  {start_date}~{end_date} → 返回 {len(result)} 条")
                return result.reset_index(drop=True)
            else:
                cache_end_dt = pd.to_datetime(cache_end)
                start_date = (cache_end_dt + datetime.timedelta(days=1)).strftime("%Y%m%d")
                print(f"📦 [缓存不足] {stock_name}  缓存 {cache_start}~{cache_end}，"
                      f"需要 {start_date}~{end_date}")

    if market == "FUT" and data_source != "akshare":
        raise ValueError("❌ 外盘期货(如 GC)仅支持 data_source='akshare'")

    # ---- 从 API 获取 ----
    pro = None
    if data_source == "akshare":
        src = "新浪外盘" if market == "FUT" else "东财"
        print(f"🚀 [akshare/{src}] 开始获取 {stock_name} 数据...")
        if market == "A":
            df = _fetch_a_share_akshare(ts_code, start_date, end_date)
        elif market == "FUT":
            df = _fetch_foreign_futures_akshare(ts_code, start_date, end_date)
        elif market == "ETF":
            print("⚠️ ETF 暂不支持 akshare，回退到 tushare")
            return fetch_data(ts_code, stock_name, market,
                             start_date, end_date, exchange, data_source="tushare")
        else:
            df = _fetch_hk_share_akshare(ts_code, start_date, end_date)
    elif data_source == "sina":
        print(f"🚀 [新浪财经] 开始获取 {stock_name} 数据...")
        if market == "A":
            df = _fetch_a_share_sina(ts_code, start_date, end_date)
        elif market == "ETF":
            print("⚠️ ETF 暂不支持新浪，回退到 tushare")
            return fetch_data(ts_code, stock_name, market,
                             start_date, end_date, exchange, data_source="tushare")
        else:
            df = _fetch_hk_share_sina(ts_code, start_date, end_date)
    elif data_source == "baostock":
        print(f"🚀 [baostock] 开始获取 {stock_name} 数据...")
        if market == "A":
            df = _fetch_a_share_baostock(ts_code, start_date, end_date)
        else:
            print("⚠️ baostock 不支持港股/ETF，回退到 akshare")
            return fetch_data(ts_code, stock_name, market,
                             start_date, end_date, exchange, data_source="akshare")
    else:
        token = os.environ.get("TUSHARE_TOKEN")
        if not token:
            raise RuntimeError("❌ 错误：未设置 TUSHARE_TOKEN 环境变量")

        ts.set_token(token)
        pro = ts.pro_api()

        print(f"🚀 [tushare] 开始获取 {stock_name} 数据...")

        if market == "A":
            df = _fetch_a_share(pro, ts_code, start_date, end_date)
        elif market == "ETF":
            df = _fetch_etf(pro, ts_code, start_date, end_date)
        else:
            df = _fetch_hk_share(pro, ts_code, start_date, end_date)

    # 检查数据完整性（外盘期货无对应交易日历，跳过）
    if market == "FUT":
        print("外盘期货：跳过交易日完整性检查")
    else:
        _check_data_completeness(df, exchange, start_date, end_date)

    # ---- 合并现金分红除权数据（外盘期货无分红） ----
    if market == "FUT":
        df["Cash_Dividend"] = 0.0
    else:
        dividends = _fetch_dividends(pro, ts_code, market, start_date, end_date)
        if not dividends.empty:
            df = df.merge(dividends, on="Date", how="left")
            df["Cash_Dividend"] = df["Cash_Dividend"].fillna(0.0)
        else:
            df["Cash_Dividend"] = 0.0
    # ----

    # ---- 写入缓存（合并已有数据，避免覆盖更宽日期范围） ----
    try:
        existing = cache_load(ts_code, market)
        if existing is not None and not existing.empty:
            combined = pd.concat([existing, df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["Date"], keep="last")
            combined = combined.sort_values("Date").reset_index(drop=True)
        else:
            combined = df.copy()
        cache_save(combined, ts_code, market)
        print(f"💾 [缓存已更新] {ts_code}")
    except Exception as e:
        print(f"⚠️ 缓存写入失败: {e}")

    print(f"✅ 缓存新增：共 {len(df)} 条记录")
    return combined


def _fetch_a_share(pro, ts_code: str, start_date: str, end_date: str,
                   data_source: str = "tushare") -> pd.DataFrame:
    """A股数据获取：日线（不复权）

    data_source: "tushare" (默认) | "akshare" (东财) | "sina" (新浪)
    """
    if data_source == "akshare":
        return _fetch_a_share_akshare(ts_code, start_date, end_date)
    if data_source == "sina":
        return _fetch_a_share_sina(ts_code, start_date, end_date)
    if data_source == "baostock":
        return _fetch_a_share_baostock(ts_code, start_date, end_date)
    # 1. 下载非复权日线
    df_daily = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
    if df_daily.empty:
        raise ValueError("❌ 错误：未获取到日线数据")

    # 2. 下载复权因子
    # try:
    #     df_adj = pro.adj_factor(ts_code=ts_code, start_date=start_date, end_date=end_date)
    # except Exception as e:
    #     print(f"⚠️ 警告：复权因子获取失败，尝试使用默认因子 1.0 ({e})")
    #     df_adj = pd.DataFrame({"trade_date": df_daily["trade_date"], "adj_factor": 1.0})

    # 3. 合并与计算前复权
    # df_adj_clean = df_adj[["trade_date", "adj_factor"]]
    # df = pd.merge(df_daily, df_adj_clean, on="trade_date", how="inner")
    df = df_daily.sort_values("trade_date")

    return _format_output(df)


def _parse_dividend_hkd(scheme: str) -> float | None:
    """从 akshare 分红方案文本中提取港币每股现金分红"""
    if not isinstance(scheme, str):
        return None
    m = re.search(r'相当于港币([\d.]+)元', scheme)
    if m:
        return float(m.group(1))
    m = re.search(r'每股派港币([\d.]+)元', scheme)
    if m:
        return float(m.group(1))
    return None


def _get_missing_dividends(df: pd.DataFrame, ts_code: str,
                           end_date: str) -> pd.DataFrame:
    """对比 Tushare adj_factor 与 akshare 分红记录，返回缺失的除权事件"""
    import akshare as ak

    # Tushare adj_factor 最后变动日期
    d = df.copy()
    d["adj_diff"] = d["adj_factor"].diff().abs()
    changes = d[d["adj_diff"] > 1e-8]
    last_adj_date = (
        str(changes["trade_date"].iloc[-1])[:10].replace("-", "")
        if not changes.empty
        else None
    )

    # akshare 分红
    ak_code = ts_code.replace(".HK", "")
    try:
        div = ak.stock_hk_dividend_payout_em(symbol=ak_code)
    except Exception:
        return pd.DataFrame()

    missing = []
    for _, r in div.iterrows():
        ex_date_str = str(r["除净日"]).replace("-", "")[:8]
        scheme = str(r["分红方案"])

        if ex_date_str > end_date:
            continue
        if "分派" in scheme and "每股派" not in scheme:
            continue
        if "送股" in scheme or "转股" in scheme:
            continue

        div_hkd = _parse_dividend_hkd(scheme)
        if div_hkd is None or div_hkd <= 0:
            continue

        is_missing = (last_adj_date is None) or (ex_date_str > last_adj_date)
        if is_missing:
            missing.append({"ex_date": ex_date_str, "dividend": div_hkd})

    if not missing:
        return pd.DataFrame()

    return pd.DataFrame(missing)


def _fetch_hk_share(pro, ts_code: str, start_date: str,
                    end_date: str, data_source: str = "tushare") -> pd.DataFrame:
    """港股数据获取：hk_daily_adj（不复权）→ akshare分红除权修正

    data_source: "tushare" (默认) | "akshare" (东财) | "sina" (新浪)
    """
    if data_source == "akshare":
        return _fetch_hk_share_akshare(ts_code, start_date, end_date)
    if data_source == "sina":
        return _fetch_hk_share_sina(ts_code, start_date, end_date)
    # 1. 下载复权日线
    df = pro.hk_daily_adj(ts_code=ts_code, start_date=start_date, end_date=end_date)
    if df.empty:
        return pd.DataFrame()

    df = df.sort_values("trade_date")

    # 2. 从 akshare 获取分红数据，对比 tushare 找出缺失的除权事件并修正
    # dividend_info = _get_missing_dividends(df, ts_code, end_date)
    # if not dividend_info.empty:
    #     modify_adj_factor(df, dividend_info)

    return _format_output(df)


def _fetch_a_share_akshare(ts_code: str, start_date: str,
                           end_date: str) -> pd.DataFrame:
    """A股：akshare (东方财富数据源) → 不复权日线"""
    import akshare as ak

    # ts_code like "000001.SZ" → akshare symbol "000001"
    symbol = ts_code.split(".")[0]

    # 使用不复权，避免 look-ahead bias
    print(f"  📡 调取 akshare stock_zh_a_hist(symbol={symbol}, 不复权) ...")
    df = ak.stock_zh_a_hist(
        symbol=symbol, period="daily",
        start_date=start_date, end_date=end_date, adjust="",  # 不复权
    )
    if df.empty:
        return pd.DataFrame()

    # 映射到标准中间格式
    df = df.rename(columns={
        "日期": "trade_date", "开盘": "open", "收盘": "close",
        "最高": "high", "最低": "low", "成交量": "vol", "成交额": "amount",
    })
    # 单位转换：成交量已为手（东财接口 2024 年起直接返回手），成交额 元→千元 (/1000)，与 tushare 对齐
    df["vol"] = pd.to_numeric(df["vol"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce") / 1000
    df["open"] = pd.to_numeric(df["open"], errors="coerce")
    df["high"] = pd.to_numeric(df["high"], errors="coerce")
    df["low"] = pd.to_numeric(df["low"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    df["adj_factor"] = 1.0   # 不复权，无复权因子
    df["ts_code"] = ts_code
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values("trade_date")

    return _format_output(df[["ts_code", "trade_date", "open", "high", "low",
                             "close", "vol", "amount"]])


def _fetch_hk_share_akshare(ts_code: str, start_date: str,
                            end_date: str) -> pd.DataFrame:
    """港股：akshare (东方财富数据源) → 不复权日线"""
    import akshare as ak

    # ts_code like "00700.HK" → akshare symbol "00700"
    symbol = ts_code.replace(".HK", "")

    # 使用不复权，避免 look-ahead bias
    print(f"  📡 调取 akshare stock_hk_hist(symbol={symbol}, 不复权) ...")
    df = ak.stock_hk_hist(
        symbol=symbol, period="daily",
        start_date=start_date, end_date=end_date, adjust="",  # 不复权
    )
    if df.empty:
        return pd.DataFrame()

    # 映射到标准中间格式
    df = df.rename(columns={
        "日期": "trade_date", "开盘": "open", "收盘": "close",
        "最高": "high", "最低": "low", "成交量": "vol", "成交额": "amount",
    })
    # 单位转换：成交量已为股（与 tushare hk_daily_adj 对齐），成交额 已为元（东财接口直接返回元）
    df["vol"] = pd.to_numeric(df["vol"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["open"] = pd.to_numeric(df["open"], errors="coerce")
    df["high"] = pd.to_numeric(df["high"], errors="coerce")
    df["low"] = pd.to_numeric(df["low"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    df["ts_code"] = ts_code
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values("trade_date")

    return _format_output(df[["ts_code", "trade_date", "open", "high", "low",
                             "close", "vol", "amount"]])


# ==================== 外盘期货（akshare 新浪数据源） ====================

def _fetch_foreign_futures_akshare(ts_code: str, start_date: str,
                                   end_date: str) -> pd.DataFrame:
    """外盘期货：akshare futures_foreign_hist → 日线（如 GC=COMEX黄金期货）

    数据源为新浪外盘期货，仅 OHLC 可靠（volume/position 恒为 0）。
    ts_code 直接传 akshare 的 symbol（如 "GC"、"SI"），市场用 market="FUT"。
    """
    import akshare as ak

    print(f"  📡 调取 akshare futures_foreign_hist(symbol={ts_code}) ...")
    df = ak.futures_foreign_hist(symbol=ts_code)
    if df.empty:
        return pd.DataFrame()

    # 新浪返回: date, open, high, low, close, volume, position, s, settlement
    df = df.rename(columns={
        "date": "trade_date", "open": "open", "high": "high",
        "low": "low", "close": "close", "volume": "vol",
    })
    for col in ["open", "high", "low", "close", "vol"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # 成交额估算：合约乘数按 100 近似（GC 每手 100 盎司）；新浪 vol 恒为 0，此处仅为格式占位
    df["amount"] = (df["close"] * df["vol"] * 100 / 1000).round(2)
    df["ts_code"] = ts_code
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df[(df["trade_date"] >= start_date) & (df["trade_date"] <= end_date)]
    df = df.sort_values("trade_date")

    return _format_output(df[["ts_code", "trade_date", "open", "high", "low",
                             "close", "vol", "amount"]])


# ==================== 新浪财经数据源 ====================

def _fetch_a_share_sina(ts_code: str, start_date: str,
                        end_date: str) -> pd.DataFrame:
    """A股：新浪财经 → 不复权日线"""
    import akshare as ak

    # ts_code like "000001.SZ" → sina symbol "sz000001"
    code = ts_code.split(".")[0]
    prefix = "sh" if ts_code.endswith(".SH") else "sz"
    symbol = f"{prefix}{code}"

    print(f"  📡 调取新浪 stock_zh_a_daily(symbol={symbol}, 不复权) ...")
    df = ak.stock_zh_a_daily(
        symbol=symbol, start_date=start_date, end_date=end_date, adjust="",
    )
    if df.empty:
        return pd.DataFrame()

    # 新浪返回到: date, open, high, low, close, volume (无 amount)
    df = df.rename(columns={
        "date": "trade_date", "open": "open", "high": "high",
        "low": "low", "close": "close", "volume": "vol",
    })
    df["open"] = pd.to_numeric(df["open"], errors="coerce")
    df["high"] = pd.to_numeric(df["high"], errors="coerce")
    df["low"] = pd.to_numeric(df["low"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["vol"] = pd.to_numeric(df["vol"], errors="coerce") / 100  # 股→手
    df["amount"] = (df["close"] * df["vol"] * 100 / 1000).round(2)  # 估算: 价×量 /1000 → 千元
    df["ts_code"] = ts_code
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values("trade_date")

    return _format_output(df[["ts_code", "trade_date", "open", "high", "low",
                              "close", "vol", "amount"]])


def _fetch_hk_share_sina(ts_code: str, start_date: str,
                         end_date: str) -> pd.DataFrame:
    """港股：新浪财经 → 不复权日线"""
    import akshare as ak

    # ts_code like "00700.HK" → sina symbol "00700"
    symbol = ts_code.replace(".HK", "")

    print(f"  📡 调取新浪 stock_hk_daily(symbol={symbol}, 不复权) ...")
    df = ak.stock_hk_daily(symbol=symbol, adjust="")
    if df.empty:
        return pd.DataFrame()

    # 新浪返回: date, open, high, low, close, volume (无 amount)
    df = df.rename(columns={
        "date": "trade_date", "open": "open", "high": "high",
        "low": "low", "close": "close", "volume": "vol",
    })
    df["open"] = pd.to_numeric(df["open"], errors="coerce")
    df["high"] = pd.to_numeric(df["high"], errors="coerce")
    df["low"] = pd.to_numeric(df["low"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["vol"] = pd.to_numeric(df["vol"], errors="coerce") / 100  # 股→手
    df["amount"] = (df["close"] * df["vol"] * 100 / 1000).round(2)  # 估算: 价×量 /1000 → 千元
    df["ts_code"] = ts_code
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values("trade_date")

    return _format_output(df[["ts_code", "trade_date", "open", "high", "low",
                              "close", "vol", "amount"]])


# ==================== baostock 数据源 ====================

def _fetch_a_share_baostock(ts_code: str, start_date: str,
                            end_date: str) -> pd.DataFrame:
    """A股：baostock → 不复权日线（不支持港股）"""
    import baostock as bs

    # ts_code like "000001.SZ" → baostock "sz.000001"
    code = ts_code.split(".")[0]
    prefix = "sh" if ts_code.endswith(".SH") else "sz"
    bs_code = f"{prefix}.{code}"

    # baostock 日期格式: YYYY-MM-DD
    start_fmt = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}"
    end_fmt = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}"
    print(f"  📡 调取 baostock query_history_k_data_plus({bs_code}, 不复权) ...")

    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"❌ baostock 登录失败: {lg.error_msg}")

    try:
        rs = bs.query_history_k_data_plus(
            bs_code,
            "date,open,high,low,close,volume,amount",
            start_date=start_fmt, end_date=end_fmt,
            frequency="d", adjustflag="3",  # 3=不复权
        )
        if rs.error_code != "0":
            raise ValueError(f"❌ baostock 查询失败: {rs.error_msg}")

        data = rs.get_data()
        if data.empty:
            raise ValueError(f"❌ baostock 未获取到 {ts_code} 数据")
    finally:
        bs.logout()

    df = data.rename(columns={
        "date": "trade_date", "open": "open", "high": "high",
        "low": "low", "close": "close", "volume": "vol", "amount": "amount",
    })
    df["open"] = pd.to_numeric(df["open"], errors="coerce")
    df["high"] = pd.to_numeric(df["high"], errors="coerce")
    df["low"] = pd.to_numeric(df["low"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["vol"] = pd.to_numeric(df["vol"], errors="coerce") / 100    # 股→手
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce") / 1000  # 元→千元
    df["ts_code"] = ts_code
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values("trade_date")

    return _format_output(df[["ts_code", "trade_date", "open", "high", "low",
                              "close", "vol", "amount"]])


def _fetch_etf(pro, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:

    df = pro.fund_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
    if df.empty:
        raise ValueError("❌ 错误：未获取到ETF日线数据")

    # 2. 下载复权因子
    try:
        df_adj = pro.fund_adj(ts_code=ts_code, start_date=start_date, end_date=end_date)
    except Exception as e:
        print(f"⚠️ 警告：复权因子获取失败，尝试使用默认因子 1.0 ({e})")
        df_adj = pd.DataFrame({"trade_date": df["trade_date"], "adj_factor": 1.0})

    # 3. 合并与计算前复权
    df_adj_clean = df_adj[["trade_date", "adj_factor"]]
    df = pd.merge(df, df_adj_clean, on="trade_date", how="inner")
    df = df.sort_values("trade_date")

    return _format_output(df)

def _format_output(df: pd.DataFrame) -> pd.DataFrame:
    """列重命名为标准格式（不复权价格）"""
    df = df.rename(columns={
        "trade_date": "Date", "open": "Open", "high": "High",
        "low": "Low", "close": "Close", "vol": "Volume", "amount": "Amount",
    })
    df["Date"] = pd.to_datetime(df["Date"])
    std_cols = ["ts_code", "Date", "Open", "High", "Low", "Close",
                "Volume", "Amount"]
    if "Cash_Dividend" in df.columns:
        std_cols.append("Cash_Dividend")
    return df[std_cols]


# ==================== 分红数据获取 ====================

def _fetch_dividends(pro, ts_code: str, market: str,
                     start_date: str, end_date: str) -> pd.DataFrame:
    """获取现金分红除权日历。返回 columns: Date, Cash_Dividend (每股现金)"""
    if market == "A":
        return _fetch_a_dividends(pro, ts_code, start_date, end_date)
    else:
        return _fetch_hk_dividends(ts_code, start_date, end_date)


def _empty_div_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["Date", "Cash_Dividend"])


def _fetch_a_dividends(pro, ts_code: str,
                       start_date: str, end_date: str) -> pd.DataFrame:
    """A股分红（tushare pro.dividend）"""
    if pro is None:
        return _empty_div_df()
    try:
        df_div = pro.dividend(ts_code=ts_code)
    except Exception:
        return _empty_div_df()
    if df_div.empty:
        return _empty_div_df()
    df_div = df_div[df_div["ex_date"].notna()].copy()
    df_div["ex_date"] = pd.to_datetime(df_div["ex_date"])
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    df_div = df_div[(df_div["ex_date"] >= start_dt) & (df_div["ex_date"] <= end_dt)]
    if df_div.empty:
        return _empty_div_df()
    return pd.DataFrame({
        "Date": df_div["ex_date"],
        "Cash_Dividend": df_div["cash_div"].astype(float),
    })


def _fetch_hk_dividends(ts_code: str,
                        start_date: str, end_date: str) -> pd.DataFrame:
    """港股分红（akshare 分红派息数据）"""
    import akshare as ak
    symbol = ts_code.replace(".HK", "")
    try:
        div = ak.stock_hk_dividend_payout_em(symbol=symbol)
    except Exception:
        return _empty_div_df()
    if div.empty:
        return _empty_div_df()
    rows = []
    for _, r in div.iterrows():
        ex_date_str = str(r["除净日"]).replace("-", "")[:8]
        scheme = str(r.get("分红方案", ""))
        div_hkd = _parse_dividend_hkd(scheme)
        if div_hkd is None or div_hkd <= 0:
            continue
        if ex_date_str < start_date or ex_date_str > end_date:
            continue
        rows.append({"Date": pd.to_datetime(ex_date_str), "Cash_Dividend": div_hkd})
    if not rows:
        return _empty_div_df()
    return pd.DataFrame(rows)


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
        # 排除已知的特殊休市日（台风/暴雨等不可抗力导致的临时休市）
        known_closed = pd.to_datetime(["2023-09-01", "2023-09-08"])
        real_missing = missing_dates[~missing_dates.isin(known_closed)]
        if len(real_missing) > 0:
            print(f"✅ 数据有误：缺失共 {len(real_missing)} 条记录")
            print("缺失的日期如下：")
            print(real_missing.strftime("%Y-%m-%d").tolist())
            raise ValueError("❌ 错误：数据有误")
        else:
            print(f"ℹ️ 仅缺失已知特殊休市日（{len(missing_dates)} 天），跳过检查")