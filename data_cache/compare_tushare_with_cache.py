#!/usr/bin/env python3
"""用 Tushare 日线数据与本地缓存（东财/akshare）做对比，差异导出 Excel

用法:
    python scripts/compare_tushare_with_cache.py                    # 对比 stock_list.txt 中所有股票
    python scripts/compare_tushare_with_cache.py A                  # 只对比 A 股
    python scripts/compare_tushare_with_cache.py HK                 # 只对比港股
    python scripts/compare_tushare_with_cache.py 601138.SH          # 对比指定股票
    python scripts/compare_tushare_with_cache.py --days 90          # 只对比最近90天

前提:
    1. 缓存中已有东财(akshare)数据（通过 update_cache.py 写入）
    2. 已设置 TUSHARE_TOKEN 环境变量

输出:
    完全一致 → 打印确认信息
    有差异   → 打印差异明细 + 导出 Excel 到 result/compare_diff_<ts_code>.xlsx
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

from core.cache import load as cache_load, list_cached
from core.data import _fetch_a_share, _fetch_hk_share

RESULT_DIR = os.path.join(ROOT, "result")
STOCK_LIST_FILE = os.path.join(ROOT, "stock_list.txt")

COMPARE_COLS = ["Open", "High", "Low", "Close", "Volume", "Amount"]

# 容差
PRICE_TOLERANCE = 0.01       # 价格：1分
VOLUME_TOLERANCE = 0.1       # 成交量（手）
AMOUNT_TOLERANCE = 1.0       # 成交额（千元）


def load_stock_list():
    """从 stock_list.txt 读取 [(ts_code, market, name), ...]"""
    stocks = []
    if os.path.exists(STOCK_LIST_FILE):
        with open(STOCK_LIST_FILE) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(",")
                if len(parts) >= 2:
                    stocks.append((
                        parts[0].strip(),
                        parts[1].strip(),
                        parts[2].strip() if len(parts) >= 3 else parts[0].strip(),
                    ))
    return stocks


def fetch_tushare(ts_code: str, market: str,
                  start_date: str, end_date: str) -> pd.DataFrame:
    """Tushare 不复权日线 → 标准化 DataFrame"""
    import tushare as ts

    token = os.environ.get("TUSHARE_TOKEN")
    if not token:
        raise RuntimeError("❌ 未设置 TUSHARE_TOKEN 环境变量")
    ts.set_token(token)
    pro = ts.pro_api()

    if market == "A":
        return _fetch_a_share(pro, ts_code, start_date, end_date)
    else:
        return _fetch_hk_share(pro, ts_code, start_date, end_date)


def compare_with_cache(ts_code: str, stock_name: str, market: str) -> dict:
    """对比 Tushare 与缓存(东财)数据，返回结果摘要"""
    result = {
        "ts_code": ts_code,
        "stock_name": stock_name,
        "market": market,
        "identical": False,
        "ts_rows": 0,
        "cache_rows": 0,
        "common_dates": 0,
        "only_ts": 0,
        "only_cache": 0,
        "diff_count": 0,
        "diff_detail": pd.DataFrame(),
        "excel_path": "",
    }

    # ── 1. 读缓存（东财数据） ──
    df_cache = cache_load(ts_code, market)
    if df_cache is None or df_cache.empty:
        print(f"  ⚠️  无缓存数据，请先运行 update_cache.py")
        result["cache_rows"] = -1
        return result

    result["cache_rows"] = len(df_cache)
    print(f"  📦 缓存(东财): {len(df_cache)} 条")

    if df_cache.empty:
        print(f"  ⚠️  缓存中无指定日期范围的数据")
        return result

    # ── 2. 拉 Tushare 数据 ──
    try:
        print(f"  🔵 拉取 Tushare ...")
        start_fetch = df_cache["Date"].min().strftime("%Y%m%d")
        end_fetch = df_cache["Date"].max().strftime("%Y%m%d")
        df_ts = fetch_tushare(ts_code, market, start_fetch, end_fetch)
        result["ts_rows"] = len(df_ts)
        print(f"  ✅ Tushare: {len(df_ts)} 条")
    except Exception as e:
        print(f"  ❌ Tushare 获取失败: {e}")
        result["ts_rows"] = -1
        return result

    # ── 3. 按日期对齐对比 ──
    cache_index = df_cache.set_index("Date")
    ts_index = df_ts.set_index("Date")

    common_dates = cache_index.index.intersection(ts_index.index)
    only_cache = cache_index.index.difference(ts_index.index)
    only_ts = ts_index.index.difference(cache_index.index)

    result["common_dates"] = len(common_dates)
    result["only_cache"] = len(only_cache)
    result["only_ts"] = len(only_ts)

    if len(only_cache) > 0:
        print(f"  📤 仅缓存有 (Tushare 缺失): {len(only_cache)} 天")
    if len(only_ts) > 0:
        print(f"  📥 仅 Tushare 有 (缓存缺失): {len(only_ts)} 天")

    # ── 4. 逐列对比共同日期 ──
    diff_rows = []
    for d in common_dates:
        row_ts = ts_index.loc[d]
        row_cache = cache_index.loc[d]
        for col in COMPARE_COLS:
            if col not in row_ts.index or col not in row_cache.index:
                continue
            v_ts = float(row_ts[col]) if pd.notna(row_ts[col]) else np.nan
            v_cache = float(row_cache[col]) if pd.notna(row_cache[col]) else np.nan

            if pd.isna(v_ts) and pd.isna(v_cache):
                continue
            if pd.isna(v_ts) or pd.isna(v_cache):
                diff_rows.append({
                    "Date": d, "Column": col,
                    "Tushare": v_ts, "Cache(东财)": v_cache,
                    "Diff": np.nan,
                })
                continue

            diff_abs = abs(v_ts - v_cache)
            if col == "Volume":
                tolerance = VOLUME_TOLERANCE
            elif col == "Amount":
                tolerance = AMOUNT_TOLERANCE
            else:
                tolerance = PRICE_TOLERANCE

            if diff_abs > tolerance:
                diff_rows.append({
                    "Date": d, "Column": col,
                    "Tushare": round(v_ts, 4),
                    "Cache(东财)": round(v_cache, 4),
                    "Diff": round(v_ts - v_cache, 4),
                })

    result["diff_count"] = len(diff_rows)

    # ── 5. 输出结果 ──
    if diff_rows or len(only_cache) > 0 or len(only_ts) > 0:
        result["identical"] = False

        # 构建导出 DataFrame
        excel_parts = []

        # 5a. 逐字段差异明细
        if diff_rows:
            df_diff = pd.DataFrame(diff_rows).sort_values(["Date", "Column"])
            excel_parts.append(("字段差异", df_diff))

            print(f"\n  ❌ 发现 {len(diff_rows)} 处字段差异：")
            for date, grp in df_diff.groupby("Date"):
                date_str = date.strftime("%Y-%m-%d")
                print(f"    📅 {date_str}:")
                for _, r in grp.iterrows():
                    print(f"       {r['Column']:>8s}  Tushare={r['Tushare']:>12.4f}  东财={r['Cache(东财)']:>12.4f}  Δ={r['Diff']:>+12.4f}")
            print()

        # 5b. 仅一方有的日期
        if len(only_cache) > 0:
            df_only_cache = (df_cache.set_index("Date")
                             .loc[only_cache]
                             .reset_index()[["Date"] + COMPARE_COLS])
            excel_parts.append(("仅缓存有", df_only_cache))
            print(f"  📤 仅缓存(东财)有的日期 ({len(only_cache)} 天):")
            print(f"     {', '.join(d.strftime('%Y-%m-%d') for d in sorted(only_cache)[:20])}"
                  + (" ..." if len(only_cache) > 20 else ""))
            print()

        if len(only_ts) > 0:
            df_only_ts = (df_ts.set_index("Date")
                          .loc[only_ts]
                          .reset_index()[["Date"] + COMPARE_COLS])
            excel_parts.append(("仅Tushare有", df_only_ts))
            print(f"  📥 仅 Tushare 有的日期 ({len(only_ts)} 天):")
            print(f"     {', '.join(d.strftime('%Y-%m-%d') for d in sorted(only_ts)[:20])}"
                  + (" ..." if len(only_ts) > 20 else ""))
            print()

        # ── 6. 导出 Excel ──
        safe_name = ts_code.replace(".", "_")
        excel_path = os.path.join(RESULT_DIR, f"compare_diff_{safe_name}.xlsx")
        os.makedirs(RESULT_DIR, exist_ok=True)

        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            for sheet_name, sheet_df in excel_parts:
                sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)

        result["excel_path"] = excel_path
        print(f"  📁 差异已导出 → {excel_path}")
    else:
        result["identical"] = True
        print(f"  ✅ 完全一致！{len(common_dates)} 个共同交易日数据完全相同")

    return result


def main():
    target_market = None
    target_code = None

    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg in ("HK", "A", "ETF"):
            target_market = arg
        elif arg == "--days" and i + 1 < len(sys.argv):
            i += 1
        elif "." in arg:
            target_code = arg
        i += 1

    print("=" * 70)
    print("🔬 Tushare vs 缓存(东财/akshare) 日线行情对比")
    print(f"🔍 对比列: {', '.join(COMPARE_COLS)}")
    print(f"📏 容差: 价格≤{PRICE_TOLERANCE}, 成交量≤{VOLUME_TOLERANCE}手, 成交额≤{AMOUNT_TOLERANCE}千元")
    print("=" * 70)

    # ── 收集待对比股票 ──
    stocks = []
    if target_code:
        for ts_code, market, name in load_stock_list():
            if ts_code == target_code:
                stocks.append((ts_code, market, name))
                break
        if not stocks:
            for m in ["A", "HK", "ETF"]:
                if cache_load(target_code, m) is not None:
                    stocks.append((target_code, m, target_code))
                    break
        if not stocks:
            print(f"❌ 未找到 {target_code}")
            return
    else:
        stocks = [(c, m, n) for c, m, n in load_stock_list()
                  if not target_market or m == target_market]

    if not stocks:
        print("⚠️ 无待对比股票")
        return

    print(f"\n📊 待对比: {len(stocks)} 只股票")

    summary = []
    for ts_code, market, name in stocks:
        print(f"\n{'─' * 60}")
        print(f"🔍 {name} ({ts_code})")
        res = compare_with_cache(ts_code, name, market)
        summary.append(res)

    # ── 汇总 ──
    print("\n" + "=" * 70)
    print("📋 对比汇总")
    print("=" * 70)

    identical = sum(1 for r in summary if r["identical"])
    differ = sum(1 for r in summary if not r["identical"] and r["ts_rows"] >= 0 and r["cache_rows"] >= 0)
    failed = sum(1 for r in summary if r["ts_rows"] < 0 or r["cache_rows"] < 0)

    for r in summary:
        if r["identical"]:
            st = "✅ 一致"
        elif r["ts_rows"] < 0 or r["cache_rows"] < 0:
            st = "⚠️ 失败"
        else:
            st = "❌ 差异"
        ts_info = f"{r['ts_rows']}条" if r["ts_rows"] >= 0 else "失败"
        c_info = f"{r['cache_rows']}条" if r["cache_rows"] >= 0 else "无缓存"
        d_info = f"{r['diff_count']}处差异" if r["diff_count"] > 0 else ""
        xl = f" → {os.path.basename(r['excel_path'])}" if r["excel_path"] else ""
        print(f"  {st}  {r['stock_name']}({r['ts_code']}): Tushare={ts_info}, 缓存={c_info}  {d_info}{xl}")

    print(f"\n🏁 一致: {identical} | 差异: {differ} | 失败: {failed} | 总计: {len(summary)}")


if __name__ == "__main__":
    main()
