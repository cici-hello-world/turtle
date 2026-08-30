#!/usr/bin/env python3
"""个股估值：百度股市通 PE(TTM)/PB 历史 → 季度选股

1. 对季度股票池并集逐个拉取 akshare stock_hk_valuation_baidu（每只 2 次调用）
2. 缓存 data_cache/VAL/{ts_code}.parquet（Date, pe_ttm, pb，ffill 对齐）
3. 失败（chartInfo 异常/新股无数据）→ _failed.csv，剔除
4. build_quarterly_selection：季度末估值表 → pe>=1 且 pb>0 → 按 pe 升序、pb 次键 → Top basket_n
"""
import os
import sys
import time

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from core.cache import load as cache_load, save as cache_save
from strategy.buffett_1957.hk_universe import build_universe

FAILED_PATH = os.path.join(PROJECT_ROOT, "data_cache", "VAL", "_failed.csv")


def _fetch_one(ts_code: str, indicator: str) -> pd.DataFrame | None:
    """百度股市通单指标全历史 → DataFrame[Date, value]；失败返回 None"""
    import akshare as ak
    symbol = ts_code.replace(".HK", "")
    try:
        df = ak.stock_hk_valuation_baidu(symbol=symbol, indicator=indicator, period="全部")
    except Exception:
        return None
    if df is None or df.empty:
        return None
    df = df.rename(columns={"date": "Date", "value": indicator})
    df["Date"] = pd.to_datetime(df["Date"])
    df[indicator] = pd.to_numeric(df[indicator], errors="coerce")
    return df[["Date", indicator]].sort_values("Date").reset_index(drop=True)


def _load_failed() -> set:
    if os.path.exists(FAILED_PATH):
        df = pd.read_csv(FAILED_PATH)
        return set(df["ts_code"]) if "ts_code" in df.columns else set()
    return set()


def _save_failed(codes: set):
    pd.DataFrame(sorted(codes), columns=["ts_code"]).to_csv(FAILED_PATH, index=False)


def collect_valuations(config, force: bool = False) -> dict:
    """拉取股票池并集估值，返回 {ts_code: DataFrame[Date, pe_ttm, pb]}"""
    universe = build_universe(config)
    codes = sorted(universe["ts_code"].unique())
    failed = set() if force else _load_failed()
    result = {}
    todo = [c for c in codes if c not in failed]
    print(f"🚀 估值拉取：池并集 {len(codes)} 只，待拉取 {len(todo)} 只")
    t0 = time.time()
    for i, code in enumerate(todo, 1):
        cached = cache_load(code, "VAL")
        if cached is not None and not cached.empty and not force:
            result[code] = cached
            continue
        pe = _fetch_one(code, "市盈率(TTM)")
        pb = _fetch_one(code, "市净率")
        if pe is None and pb is None:
            failed.add(code)
        else:
            merged = None
            if pe is not None:
                merged = pe
            if pb is not None:
                merged = pb if merged is None else merged.merge(pb, on="Date", how="outer")
            merged = merged.rename(columns={"市盈率(TTM)": "pe_ttm", "市净率": "pb"})
            for col in ["pe_ttm", "pb"]:          # 单指标失败时补齐缺失列
                if col not in merged.columns:
                    merged[col] = np.nan
            merged = merged.sort_values("Date")
            merged[["pe_ttm", "pb"]] = merged[["pe_ttm", "pb"]].ffill()
            cache_save(merged, code, "VAL")
            result[code] = merged
        if i % 100 == 0:
            print(f"  [{i}/{len(todo)}] 耗时 {time.time()-t0:.0f}s")
    if failed:
        all_failed = _load_failed() | failed
        _save_failed(all_failed)
        print(f"⚠️ {len(failed)} 只失败（已记入 _failed.csv）")
    print(f"✅ 估值缓存完成 {len(result)} 只")
    return result


def _valuation_at(code: str, val_df: pd.DataFrame, qe: pd.Timestamp) -> dict | None:
    """季度末时点估值：≤qe 的最后一行（无 → None）"""
    sub = val_df[val_df["Date"] <= qe]
    if sub.empty:
        return None
    row = sub.iloc[-1]
    return {"ts_code": code, "pe_ttm": row.get("pe_ttm", np.nan),
            "pb": row.get("pb", np.nan)}


def build_quarterly_selection(config, valuations: dict | None = None,
                              force: bool = False) -> dict:
    """返回 {季度末 Timestamp: DataFrame[ts_code, pe_ttm, pb]}（Top basket_n 等权候选）"""
    if valuations is None:
        valuations = collect_valuations(config, force=force)
    universe = build_universe(config)
    failed = _load_failed()
    selections = {}
    for qe, group in universe.groupby("Date"):
        qe = pd.Timestamp(qe)
        rows = []
        for code in group["ts_code"]:
            if code in failed or code not in valuations:
                continue
            r = _valuation_at(code, valuations[code], qe)
            if r is not None:
                rows.append(r)
        if not rows:
            selections[qe] = pd.DataFrame(columns=["ts_code", "pe_ttm", "pb"])
            continue
        df = pd.DataFrame(rows)
        df = df[(df["pe_ttm"] >= 1) & (df["pb"] > 0)]          # 剔除亏损/异常
        df = df.sort_values(["pe_ttm", "pb"], ascending=[True, True])
        selections[qe] = df.head(config.basket_n).reset_index(drop=True)
    return selections


def build_filtered_selection(config, min_turnover: float = 1e8,
                             min_mcap: float = 1e11) -> dict:
    """流动性+市值双重过滤后的季度选股：
    1. 季度末 5 日均成交额 ≥ min_turnover（港元）
    2. 季度末总市值 ≥ min_mcap（元，hk_daily_adj.total_mv，ffill ≤ 季度末）
    3. 估值过滤 pe_ttm≥1 且 pb>0，按 pe 升序、pb 次键取 Top basket_n
    返回 {季度末 Timestamp: DataFrame[ts_code, pe_ttm, pb]}"""
    from strategy.buffett_1957.hk_universe import collect_market_caps

    # 1. 流动性过滤（复用全市场快照缓存）
    liq = cache_load("liquidity", "VAL")
    if liq is None or liq.empty:
        raise RuntimeError("❌ 需先运行 build_universe 生成流动性缓存")
    liq = liq[liq["Date"] >= pd.to_datetime(config.start_date)]
    avg = liq.groupby(["Date", "ts_code"])["amount"].mean().reset_index()
    cand = avg[avg["amount"] >= min_turnover].copy()
    cand["Date"] = pd.to_datetime(cand["Date"])
    print(f"流动性过滤后候选: {cand['ts_code'].nunique()} 只 / {cand['Date'].nunique()} 季度")

    # 2. 市值数据（候选并集一次性拉取）
    caps = collect_market_caps(cand["ts_code"].unique(), config.start_date, config.end_date)
    # 市值序列 ffill 到季度末查询
    cap_series = {c: v.set_index("Date")["total_mv"] for c, v in caps.items()}

    # 3. 估值（复用缓存）
    valuations = collect_valuations(config)
    failed = _load_failed()

    selections = {}
    for qe, group in cand.groupby("Date"):
        qe = pd.Timestamp(qe)
        rows = []
        for code in group["ts_code"]:
            if code in failed or code not in valuations or code not in cap_series:
                continue
            mv_s = cap_series[code]
            mv_s = mv_s[mv_s.index <= qe]
            if mv_s.empty or mv_s.iloc[-1] < min_mcap:
                continue
            r = _valuation_at(code, valuations[code], qe)
            if r is not None:
                rows.append(r)
        if not rows:
            selections[qe] = pd.DataFrame(columns=["ts_code", "pe_ttm", "pb"])
            continue
        df = pd.DataFrame(rows)
        df = df[(df["pe_ttm"] >= 1) & (df["pb"] > 0)]
        df = df.sort_values(["pe_ttm", "pb"], ascending=[True, True])
        selections[qe] = df.head(config.basket_n).reset_index(drop=True)
    print(f"筛选后季度选股: {len(selections)} 个季度")
    return selections


if __name__ == "__main__":
    from strategy.buffett_1957.config import Buffett1957Config
    cfg = Buffett1957Config()
    vals = collect_valuations(cfg)
    sel = build_quarterly_selection(cfg, vals)
    print(f"季度选股: {len(sel)} 个季度")
    for qe in [pd.Timestamp("2016-03-31"), pd.Timestamp("2020-03-31"),
               pd.Timestamp("2024-03-31"), pd.Timestamp("2026-06-30")]:
        if qe in sel:
            s = sel[qe]
            print(f"\n{qe.date()} 持仓 {len(s)} 只:")
            print(s.head(10).to_string(index=False))
