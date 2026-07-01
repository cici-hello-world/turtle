#!/usr/bin/env python3
"""日线数据缓存更新脚本 —— 每天定时运行，增量拉取新数据追加到本地缓存

用法:
    python scripts/update_cache.py                # 更新所有已缓存股票
    python scripts/update_cache.py HK             # 只更新港股
    python scripts/update_cache.py A              # 只更新 A 股
    python scripts/update_cache.py --all          # 更新所有股票（含 stock_list.txt）

定时运行（crontab）:
    # 每个交易日 18:00 更新
    0 18 * * 1-5 cd /path/to/turtle && python scripts/update_cache.py >> logs/update.log 2>&1

股票列表文件（stock_list.txt）:
    # 每行: ts_code,market,stock_name
    00700.HK,HK,腾讯控股
    000001.SZ,A,平安银行
"""

import os
import sys
import pandas as pd
from datetime import datetime, timedelta

# 项目根目录
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

from core.cache import load, save, list_cached, last_cached_date

DATA_SOURCE = os.environ.get("TURTLE_DATA_SOURCE", "akshare")
STOCK_LIST_FILE = os.path.join(ROOT, "stock_list.txt")


def load_stock_list() -> list[tuple[str, str, str]]:
    """从 stock_list.txt 读取关注股票列表，返回 [(ts_code, market, name), ...]"""
    stocks = []
    if os.path.exists(STOCK_LIST_FILE):
        with open(STOCK_LIST_FILE) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(",")
                if len(parts) >= 2:
                    ts_code = parts[0].strip()
                    market = parts[1].strip()
                    name = parts[2].strip() if len(parts) >= 3 else ts_code
                    stocks.append((ts_code, market, name))
    return stocks


def update_one(ts_code: str, stock_name: str, market: str) -> bool:
    """增量更新单只股票，返回 True 表示有更新"""
    last_date = last_cached_date(ts_code, market)
    if last_date is None:
        # 无缓存，全量拉取,可修改start_date和end_date
        start_date = "20210812"
        end_date = (pd.Timestamp.today() - pd.DateOffset(years=5)).strftime("%Y%m%d")
        print(f"  🆕 {ts_code} {stock_name}: 无缓存，全量拉取")
    else:
        # 增量：从最后日期+1天开始
        start_date = (last_date + timedelta(days=1)).strftime("%Y%m%d")
        end_date = datetime.today().strftime("%Y%m%d")
        if datetime.now().hour < 20:
            # 晚上8点之后才拉取当前的，否则值拉取到前一天
            end_date = (datetime.today() - timedelta(days=1)).strftime("%Y%m%d")
        if start_date > end_date:
            print(f"  ✅ {ts_code} {stock_name}: 已是最新 ({last_date.strftime('%Y-%m-%d')})")
            return False
        print(f"  📡 {ts_code} {stock_name}: 增量 {start_date} ~ {end_date}")

    # 重要：use_cache=False 避免递归读缓存
    from core.data import fetch_data

    exchange_map = {"A": "XSHG", "HK": "XHKG", "ETF": "XSHG"}
    exchange = exchange_map.get(market, "XSHG")

    try:
        new_df = fetch_data(
            ts_code=ts_code,
            stock_name=stock_name,
            market=market,
            start_date=start_date,
            end_date=end_date,
            exchange=exchange,
            data_source=DATA_SOURCE,
            use_cache=False,
        )
    except Exception as e:
        print(f"  ❌ {ts_code} 获取失败: {e}")
        return False

    # 与旧缓存合并
    existing = load(ts_code, market)
    if existing is not None and not existing.empty:
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["Date"], keep="last")
        combined = combined.sort_values("Date").reset_index(drop=True)
    else:
        combined = new_df

    save(combined, ts_code, market)
    print(f"  💾 {ts_code}: {len(existing) if existing is not None else 0} → {len(combined)} 条")
    return True


def main():
    target_market = None
    use_all = False

    for arg in sys.argv[1:]:
        if arg in ("HK", "A", "ETF"):
            target_market = arg
        elif arg == "--all":
            use_all = True

    print(f"📡 数据源: {DATA_SOURCE}")
    print(f"🕐 时间: {datetime.today().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    # 收集待更新股票
    to_update: list[tuple[str, str, str]] = []

    if use_all or not target_market:
        # 从 stock_list.txt 读取
        for ts_code, market, name in load_stock_list():
            if target_market and market != target_market:
                continue
            to_update.append((ts_code, market, name))

    # 补充：已缓存但不在列表中的股票
    for ts_code in list_cached(target_market):
        # 推断 market：从缓存目录结构
        for m in ["A", "HK", "ETF"]:
            cached = load(ts_code, m)
            if cached is not None and not cached.empty:
                if (ts_code, m, ts_code) not in to_update:
                    to_update.append((ts_code, m, ts_code))
                break

    if not to_update:
        print("⚠️ 无待更新股票。请创建 stock_list.txt 或先手动拉取数据。")
        return

    print(f"📊 待更新: {len(to_update)} 只股票\n")

    updated = 0
    for ts_code, market, name in to_update:
        if update_one(ts_code, name, market):
            updated += 1

    print(f"\n{'=' * 60}")
    print(f"✅ 完成: {updated}/{len(to_update)} 只有更新")


if __name__ == "__main__":
    main()
