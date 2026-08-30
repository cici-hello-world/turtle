#!/usr/bin/env python3
"""恒指估值数据链

主路径：乐咕乐股非官方 API → 恒指等权静态 PE 月序列（1973→今）
  URL  /api/stockdata/index-basic-pe?token=md5(今天)&indexCode=HSI
  免费档仅 addLyrPe 有真实值；接口偶发超时 → 退避重试 3 次
兜底路径：恒指点位历史分位（tushare index_global 按年分页）作为估值代理
缓存：data_cache/IDX/HSI_PE.parquet、data_cache/IDX/HSI.parquet
"""
import hashlib
import os
import sys
import time
from datetime import datetime

import pandas as pd
import requests

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from core.cache import load as cache_load, save as cache_save

LEGU_URL = "https://legulegu.com/api/stockdata/index-basic-pe"
COOKIE_URL = "https://legulegu.com/stockdata/sz50-ttm-lyr"
BACKOFF = (1, 3, 9)


def _token() -> str:
    """乐咕乐股 token = md5(今天日期)（与 akshare 内部 JS hex 函数等价）"""
    return hashlib.md5(datetime.now().date().isoformat().encode()).hexdigest()


def _fetch_legu(max_retries: int = 3) -> pd.DataFrame | None:
    """拉取恒指等权静态 PE 月序列，失败退避重试；返回 DataFrame[Date, addLyrPe]"""
    from akshare.stock_feature.stock_a_indicator import get_cookie_csrf
    last_err = None
    for i in range(max_retries):
        try:
            r = requests.get(LEGU_URL,
                             params={"token": _token(), "indexCode": "HSI"},
                             timeout=(10, 60),
                             **get_cookie_csrf(url=COOKIE_URL))
            data_json = r.json()
            if data_json.get("data"):
                df = pd.DataFrame(data_json["data"])
                df = df.rename(columns={"date": "Date"})[["Date", "addLyrPe"]]
                df["Date"] = pd.to_datetime(df["Date"])
                df["addLyrPe"] = pd.to_numeric(df["addLyrPe"], errors="coerce")
                df = df.sort_values("Date").reset_index(drop=True)
                return df
            last_err = f"无 data 字段: {str(data_json)[:120]}"
        except Exception as e:
            last_err = repr(e)
        if i < max_retries - 1:
            time.sleep(BACKOFF[min(i, len(BACKOFF) - 1)])
    print(f"⚠️ legulegu 拉取失败: {last_err}")
    return None


def load_hsi_pe(force: bool = False) -> tuple[pd.DataFrame, str]:
    """返回 (PE 月序列, 来源标签)。缓存优先；拉取失败时返回空表。"""
    if not force:
        cached = cache_load("HSI_PE", "IDX")
        if cached is not None and not cached.empty:
            print(f"📦 [缓存命中] HSI_PE  {len(cached)} 行 "
                  f"{cached['Date'].min().date()} ~ {cached['Date'].max().date()}")
            return cached, "cache"
    df = _fetch_legu()
    if df is not None and not df.empty:
        cache_save(df, "HSI_PE", "IDX")
        print(f"✅ HSI_PE 已缓存 {len(df)} 行")
        return df, "legulegu"
    return pd.DataFrame(columns=["Date", "addLyrPe"]), "failed"


def load_hsi_price() -> pd.DataFrame:
    """恒指日线（tushare index_global 大窗口分页），缓存 data_cache/IDX/HSI.parquet。
    同时作为组合回测的主交易日历与买入持有基准。
    缓存只覆盖部分年份时，增量补齐缺失区间。"""
    cached = cache_load("HSI", "IDX")
    if cached is not None and not cached.empty:
        need_1994 = cached["Date"].min().year > 1994
        if not need_1994:
            print(f"📦 [缓存命中] HSI 日线 {len(cached)} 行")
            return cached

    import tushare as ts
    token = os.environ.get("TUSHARE_TOKEN")
    if not token:
        raise RuntimeError("❌ 未设置 TUSHARE_TOKEN 环境变量")
    ts.set_token(token)
    pro = ts.pro_api()

    frames = []
    # index_global 单次上限 4000 行且限频 10 次/分钟 → 两个大窗口分页 + 限频重试
    for start_y, end_y in [(1994, 2010), (2011, datetime.now().year)]:
        if cached is not None and not cached.empty:
            covered = (cached["Date"].min().year <= start_y
                       and cached["Date"].max().year >= end_y)
            if covered:
                continue
        df = None
        for attempt in range(3):
            try:
                df = pro.index_global(ts_code="HSI",
                                      start_date=f"{start_y}0101",
                                      end_date=f"{end_y}1231")
                break
            except Exception as ex:
                print(f"  ⚠️ index_global {start_y}-{end_y} 失败: {ex}，等待重试")
                time.sleep(30)
        if df is not None and not df.empty:
            frames.append(df)
        time.sleep(7)
    if not frames:
        if cached is not None and not cached.empty:
            print(f"⚠️ 增量拉取失败，沿用现有缓存（{cached['Date'].min().date()} 起）")
            return cached
        raise RuntimeError("❌ index_global(HSI) 未获取到数据")
    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={"trade_date": "Date"})[["Date", "close"]]
    df["Date"] = pd.to_datetime(df["Date"])
    df["Close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["Close"]).sort_values("Date").reset_index(drop=True)
    if cached is not None and not cached.empty:
        df = pd.concat([cached, df], ignore_index=True)
        df = df.drop_duplicates(subset=["Date"], keep="last").sort_values("Date")
        df = df.reset_index(drop=True)
    cache_save(df[["Date", "Close"]], "HSI", "IDX")
    print(f"✅ HSI 日线已缓存 {len(df)} 行")
    return df[["Date", "Close"]]


def get_monthly_weights(config, pe_df: pd.DataFrame | None = None,
                        source: str = "cache") -> tuple[pd.Series, pd.Series, str]:
    """返回 (月度权重 Series, 估值分位 Series, 来源标签)。

    PE 可用 → 用 PE 分位；PE 不可用 → 兜底恒指点位分位（标签 fallback_price）。
    """
    from strategy.buffett_1957.allocation import expanding_percentile, monthly_weights

    if pe_df is not None and not pe_df.empty:
        pe = pe_df.set_index("Date")["addLyrPe"].dropna()
        # 分位数必须用全历史计算（无前视），输出时再截取回测区间
        weights_all = monthly_weights(pe, config.lo, config.hi, config.w_min,
                                      config.min_history)
        pct_all = expanding_percentile(pe)
        mask = pe.index >= pd.to_datetime(config.start_date)
        return weights_all[mask], pct_all[mask], \
            source if pe_df is not None and source != "failed" else "legulegu"

    # 兜底：恒指点位分位
    hsi = load_hsi_price()
    price = hsi.set_index("Date")["Close"]
    # 月度末采样后计算分位（全历史，输出再截取）
    month_end = price.resample("ME").last().dropna()
    weights_all = monthly_weights(month_end, config.lo, config.hi, config.w_min,
                                  config.min_history)
    pct_all = expanding_percentile(month_end)
    mask = month_end.index >= pd.to_datetime(config.start_date)
    return weights_all[mask], pct_all[mask], "fallback_price"


if __name__ == "__main__":
    """自检：拉取 PE → 打印抽样分位"""
    pe_df, src = load_hsi_pe()
    print(f"来源: {src}, 行数: {len(pe_df)}")
    if not pe_df.empty:
        from strategy.buffett_1957.allocation import expanding_percentile
        pe = pe_df.set_index("Date")["addLyrPe"].dropna()
        pct = expanding_percentile(pe)
        for label in ["2018-12-31", "2020-03-31", "2021-01-31",
                      "2022-10-31", "2024-01-31", "2026-07-31"]:
            idx = pe.index[pe.index <= pd.to_datetime(label)][-1]
            print(f"  {label} → PE={pe.loc[idx]:.1f} 分位={pct.loc[idx]:.2f}")
