"""本地行情数据缓存 —— Parquet 格式，按 market/ts_code 组织

目录结构:
    data_cache/
    ├── A/
    │   ├── 000001.SZ.parquet
    │   └── ...
    ├── HK/
    │   ├── 00700.HK.parquet
    │   └── ...
    └── ETF/
        └── ...
"""

import os
import pandas as pd

CACHE_ROOT = os.environ.get(
    "TURTLE_CACHE_DIR",
    os.path.join(os.path.dirname(__file__), "..", "data_cache"),
)


def _cache_dir(market: str) -> str:
    m = {"A": "A", "HK": "HK", "ETF": "ETF"}
    return os.path.join(CACHE_ROOT, m.get(market, market))


def _cache_path(ts_code: str, market: str) -> str:
    return os.path.join(_cache_dir(market), f"{ts_code}.parquet")


# ==================== 读写接口 ====================

def load(ts_code: str, market: str) -> pd.DataFrame | None:
    """从缓存读取，文件不存在返回 None"""
    path = _cache_path(ts_code, market)
    if os.path.exists(path):
        df = pd.read_parquet(path)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
        return df
    return None


def save(df: pd.DataFrame, ts_code: str, market: str):
    """写入缓存"""
    path = _cache_path(ts_code, market)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)


def get_date_range(ts_code: str, market: str) -> tuple | None:
    """返回 (min_date, max_date)，无缓存返回 None"""
    df = load(ts_code, market)
    if df is not None and not df.empty:
        return df["Date"].min(), df["Date"].max()
    return None


# ==================== 缓存列表 ====================

def list_cached(market: str | None = None) -> list[str]:
    """列出已缓存的 ts_code 列表"""
    result = []
    markets = [market] if market else ["A", "HK", "ETF"]
    for m in markets:
        d = _cache_dir(m)
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.endswith(".parquet"):
                result.append(f.replace(".parquet", ""))
    return sorted(result)


def last_cached_date(ts_code: str, market: str) -> pd.Timestamp | None:
    """返回缓存中最新日期"""
    dr = get_date_range(ts_code, market)
    return dr[1] if dr else None


def cache_info() -> pd.DataFrame:
    """返回所有缓存文件的信息汇总"""
    rows = []
    for ts_code in list_cached():
        for m in ["A", "HK", "ETF"]:
            path = _cache_path(ts_code, m)
            if os.path.exists(path):
                df = load(ts_code, m)
                rows.append({
                    "ts_code": ts_code,
                    "market": m,
                    "rows": len(df),
                    "start": df["Date"].min().strftime("%Y-%m-%d"),
                    "end": df["Date"].max().strftime("%Y-%m-%d"),
                    "size_kb": round(os.path.getsize(path) / 1024, 1),
                })
                break
    return pd.DataFrame(rows).sort_values("ts_code") if rows else pd.DataFrame()
