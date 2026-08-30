#!/usr/bin/env python3
"""成员日线：不复权价格 + adj_factor 差分推算现金分红

pro.hk_daily_adj 返回不复权 OHLC + adj_factor（分红/拆股累积因子）。
每股现金分红近似：adj_factor 日比值 < 0.9999 视为除权事件，
D ≈ prev_close × (1 − adj_t/adj_{t-1})（拆股罕见，MVP 不区分）。
缓存：data_cache/BUF/{ts_code}.parquet（与 data_cache/HK/ 现有股票缓存隔离）
"""
import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from core.cache import load as cache_load, save as cache_save


def _fetch_one(ts_code: str, start: str, end: str) -> pd.DataFrame | None:
    import tushare as ts
    token = os.environ.get("TUSHARE_TOKEN")
    if not token:
        raise RuntimeError("❌ 未设置 TUSHARE_TOKEN 环境变量")
    ts.set_token(token)
    pro = ts.pro_api()
    try:
        df = pro.hk_daily_adj(ts_code=ts_code, start_date=start, end_date=end)
    except Exception as ex:
        print(f"  ⚠️ {ts_code} 日线拉取失败: {ex}")
        return None
    if df is None or df.empty:
        return None
    df = df.sort_values("trade_date").reset_index(drop=True)
    out = pd.DataFrame({
        "Date": pd.to_datetime(df["trade_date"]),
        "Open": pd.to_numeric(df["open"], errors="coerce"),
        "High": pd.to_numeric(df["high"], errors="coerce"),
        "Low": pd.to_numeric(df["low"], errors="coerce"),
        "Close": pd.to_numeric(df["close"], errors="coerce"),
    })
    if "adj_factor" in df.columns:
        adj = pd.to_numeric(df["adj_factor"], errors="coerce")
        ratio = adj / adj.shift(1)
        prev_close = out["Close"].shift(1)
        div = np.where(ratio < 0.9999, prev_close * (1 - ratio), 0.0)
        out["Cash_Dividend"] = pd.Series(div).round(4).fillna(0.0)
    else:
        out["Cash_Dividend"] = 0.0
    return out


def collect_prices(codes, start: str, end: str, force: bool = False) -> dict:
    """返回 {ts_code: DataFrame[Date, Open..Close, Cash_Dividend]}。
    缓存起点晚于 start（如试跑是 2016 起）时自动重拉补全。"""
    start_dt = pd.to_datetime(start)
    result = {}
    todo = []
    for c in sorted(set(codes)):
        cached = cache_load(c, "BUF")
        if (cached is not None and not cached.empty and not force
                and cached["Date"].min() <= start_dt):
            result[c] = cached
        else:
            todo.append(c)
    print(f"🚀 成员日线拉取：{len(todo)} 只待拉取")
    for i, c in enumerate(todo, 1):
        df = _fetch_one(c, start, end)
        if df is not None:
            cache_save(df, c, "BUF")
            result[c] = df
        if i % 100 == 0:
            print(f"  [{i}/{len(todo)}]")
    print(f"✅ 成员日线可用 {len(result)} 只")
    return result
