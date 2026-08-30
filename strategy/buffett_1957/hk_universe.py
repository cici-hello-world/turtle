#!/usr/bin/env python3
"""季度股票池：全市场流动性快照 Top-N

1. pro.hk_basic(list_status='L') 取当前上市名单（MVP 接受幸存者偏差）
2. 每季度末最后 liquidity_days 个交易日 pro.hk_daily(trade_date=) 全市场成交额
3. 按季度平均成交额取 Top universe_n → 季度股票池
缓存：data_cache/VAL/liquidity.parquet（原始快照）、data_cache/VAL/universe.parquet（季度池）
"""
import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from core.cache import load as cache_load, save as cache_save


def load_hk_basic() -> pd.DataFrame:
    """港股当前上市名单（缓存 data_cache/VAL/hk_basic.parquet）"""
    cached = cache_load("hk_basic", "VAL")
    if cached is not None and not cached.empty:
        print(f"📦 [缓存命中] hk_basic {len(cached)} 只")
        return cached
    import tushare as ts
    token = os.environ.get("TUSHARE_TOKEN")
    if not token:
        raise RuntimeError("❌ 未设置 TUSHARE_TOKEN 环境变量")
    ts.set_token(token)
    pro = ts.pro_api()
    df = pro.hk_basic(list_status="L")
    df = df[["ts_code", "name", "trade_unit", "market"]].copy()
    df["trade_unit"] = pd.to_numeric(df["trade_unit"], errors="coerce").fillna(100).astype(int)
    cache_save(df, "hk_basic", "VAL")
    print(f"✅ hk_basic 已缓存 {len(df)} 只")
    return df


def _quarter_ends(start: str, end: str) -> list[pd.Timestamp]:
    """季度末日期列表（含起止月）"""
    s = pd.to_datetime(start)
    e = pd.to_datetime(end)
    dates = pd.date_range(s, e, freq="QE")
    return [d for d in dates]


def _quarter_liquidity_dates(trading_days: pd.DatetimeIndex, quarter_end: pd.Timestamp,
                             n_days: int) -> list[pd.Timestamp]:
    """季度末（含当日）往前 n_days 个交易日。
    交易日历用恒指日线（exchange_calendars 的 XHKG 只覆盖 2006-08 之后）。"""
    days = trading_days[trading_days <= quarter_end]
    return list(days[-n_days:])


def collect_liquidity(config, force: bool = False) -> pd.DataFrame:
    """季度末快照成交额，返回 DataFrame[Date, ts_code, amount]"""
    cached = cache_load("liquidity", "VAL")
    if cached is not None and not cached.empty and not force:
        print(f"📦 [缓存命中] liquidity {len(cached)} 行")
        return cached

    import tushare as ts
    token = os.environ.get("TUSHARE_TOKEN")
    if not token:
        raise RuntimeError("❌ 未设置 TUSHARE_TOKEN 环境变量")
    ts.set_token(token)
    pro = ts.pro_api()

    hsi = cache_load("HSI", "IDX")
    if hsi is None or hsi.empty:
        raise RuntimeError("❌ 需先缓存恒指日线（运行 hsi_pe.load_hsi_price）")
    trading_days = pd.DatetimeIndex(hsi["Date"])

    rows = []
    quarters = _quarter_ends(config.start_date, config.end_date)
    print(f"🚀 拉取 {len(quarters)} 个季度 × {config.liquidity_days} 天全市场成交额 ...")
    for qi, qe in enumerate(quarters, 1):
        for d in _quarter_liquidity_dates(trading_days, qe, config.liquidity_days):
            ds = d.strftime("%Y%m%d")
            try:
                df = pro.hk_daily(trade_date=ds)
            except Exception as ex:
                print(f"  ⚠️ {ds} 拉取失败: {ex}")
                continue
            if df is None or df.empty:
                continue
            df = df[["ts_code", "amount"]].copy()
            df["Date"] = qe
            df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
            rows.append(df.dropna(subset=["amount"]))
        if qi % 20 == 0:
            print(f"  [{qi}/{len(quarters)}] 季度完成")
    if not rows:
        raise RuntimeError("❌ 流动性快照全部失败")
    liq = pd.concat(rows, ignore_index=True)
    cache_save(liq, "liquidity", "VAL")
    print(f"✅ liquidity 已缓存 {len(liq)} 行")
    return liq


def build_universe(config, force: bool = False) -> pd.DataFrame:
    """季度股票池：每季度末平均成交额 Top-N，返回 DataFrame[Date, ts_code]"""
    cached = cache_load("universe", "VAL")
    if cached is not None and not cached.empty and not force:
        print(f"📦 [缓存命中] universe {len(cached)} 行")
        return cached

    basic = load_hk_basic()
    liq = collect_liquidity(config, force=force)   # force 透传，避免复用旧区间快照
    valid_codes = set(basic["ts_code"])
    liq = liq[liq["ts_code"].isin(valid_codes)]
    avg = liq.groupby(["Date", "ts_code"])["amount"].mean().reset_index()
    avg = avg.sort_values(["Date", "amount"], ascending=[True, False])
    top = avg.groupby("Date").head(config.universe_n).reset_index(drop=True)
    cache_save(top[["Date", "ts_code"]], "universe", "VAL")
    print(f"✅ universe 已缓存 {len(top)} 行（{top['Date'].nunique()} 个季度）")
    return top[["Date", "ts_code"]]


def collect_market_caps(codes, start: str, end: str, force: bool = False) -> dict:
    """{ts_code: DataFrame[Date, total_mv]}（总市值，元）。
    数据源 hk_daily_adj.total_mv；缓存 data_cache/VAL/{ts_code}_mv.parquet。"""
    import tushare as ts
    token = os.environ.get("TUSHARE_TOKEN")
    if not token:
        raise RuntimeError("❌ 未设置 TUSHARE_TOKEN 环境变量")
    ts.set_token(token)
    pro = ts.pro_api()

    result = {}
    todo = []
    for c in sorted(set(codes)):
        cached = cache_load(f"{c}_mv", "VAL")
        if cached is not None and not cached.empty and not force \
                and cached["Date"].min() <= pd.to_datetime(start):
            result[c] = cached
        else:
            todo.append(c)
    print(f"🚀 市值拉取：{len(todo)} 只待拉取")
    for i, c in enumerate(todo, 1):
        try:
            df = pro.hk_daily_adj(ts_code=c, start_date=start, end_date=end)
        except Exception as ex:
            print(f"  ⚠️ {c} 失败: {ex}")
            continue
        if df is None or df.empty:
            continue
        out = pd.DataFrame({
            "Date": pd.to_datetime(df["trade_date"]),
            "total_mv": pd.to_numeric(df["total_mv"], errors="coerce"),
        }).dropna().sort_values("Date").reset_index(drop=True)
        cache_save(out, f"{c}_mv", "VAL")
        result[c] = out
        if i % 100 == 0:
            print(f"  [{i}/{len(todo)}]")
    print(f"✅ 市值缓存完成 {len(result)} 只")
    return result


if __name__ == "__main__":
    from strategy.buffett_1957.config import Buffett1957Config
    cfg = Buffett1957Config()
    uni = build_universe(cfg)
    print(uni.groupby("Date").size().head(5))
    print(uni.groupby("Date").size().tail(5))
