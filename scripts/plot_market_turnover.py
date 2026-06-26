#!/usr/bin/env python3
"""港股市场 + 5只个股 每日成交额曲线

- 恒生指数日成交额双源对比：
    · 新浪：akshare stock_hk_index_daily_sina(symbol="HSI")
    · 东财：akshare stock_hk_index_daily_em(symbol="HSI")
         （若 akshare spot API 不通，回退直连 EastMoney K线 API）
  两份数据合并写入同一缓存 CSV，方便对比。
- 个股日成交额：本地缓存直接读取
- 一张图：双源市场成交额 + 个股折线
"""

import sys
import os
import numpy as np
import pandas as pd
import akshare as ak
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CACHE_PATH = os.path.join("../data_cache", "hk_market_turnover.csv")


def _load_cache() -> pd.DataFrame | None:
    """读取已有缓存"""
    if os.path.exists(CACHE_PATH):
        cached = pd.read_csv(CACHE_PATH, parse_dates=["Date"])
        if not cached.empty:
            return cached
    return None


def fetch_hk_turnover_sina(start_date: str, end_date: str) -> pd.DataFrame:
    """新浪接口：akshare stock_hk_index_daily_sina(symbol="HSI")
    返回 Date, Turnover_sina (亿港元)
    """
    print(f"🚀 [新浪 HSI] 拉取恒生指数日成交额...")
    df = ak.stock_hk_index_daily_sina(symbol="HSI")
    df = df[["date", "amount"]].copy()
    df["Date"] = pd.to_datetime(df["date"])
    df["Turnover_sina"] = df["amount"] / 1e8  # 港元 → 亿港元
    df = df.sort_values("Date").reset_index(drop=True)
    print(f"   新浪: {df['Date'].min().date()}→{df['Date'].max().date()}  "
          f"共{len(df)}天  最新¥{df['Turnover_sina'].iloc[-1]:.0f}亿")
    return df[["Date", "Turnover_sina"]]


def fetch_hk_turnover_em_ak(start_date: str, end_date: str) -> pd.DataFrame:
    """东财接口(akshare)：ak.stock_hk_index_daily_em(symbol="HSI")
    返回 Date, Turnover_em_ak (亿港元)

    注：akshare 内部需先调 spot API 拿 symbol→code 映射表，
    若网络不通则绕过，直接用已知的 HSI→100 构造请求。
    """
    import time

    print(f"🚀 [东财·akshare HSI] 拉取恒生指数日成交额...")

    # 尝试调用 akshare 原生函数
    for attempt in range(3):
        try:
            df = ak.stock_hk_index_daily_em(symbol="HSI")
            # akshare 返回列: date, open, high, low, latest（无 amount）
            # 需要补 amount，改用下方 requests 直连
            if "amount" not in df.columns and "Turnover_em_ak" not in df.columns:
                raise ValueError("akshare 东财接口不含成交额字段，改用直连")
            print(f"   akshare东财: {df['date'].min()}→{df['date'].max()}  "
                  f"共{len(df)}天")
            return df
        except Exception as e:
            if attempt < 2:
                wait = 2 ** attempt
                print(f"   ⚠️ akshare东财失败，{wait}s 后重试...")
                time.sleep(wait)

    # 回退：绕过 akshare 的 spot API，直接用已知映射构造请求
    print(f"   🔄 回退：绕过 spot API，直连东财 K线 (secid=100.HSI)")
    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    params = {
        "secid": "100.HSI",
        "klt": "101",
        "fqt": "1",
        "lmt": "1000000",
        "end": "20500000",
        "iscca": "1",
        "fields1": "f1,f2,f3,f4,f5,f6,f7,f8",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64",
        "ut": "f057cbcbce2a86e2866ab8877db1d059",
        "forcect": "1",
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://quote.eastmoney.com/",
    }
    r = requests.get(url, params=params, headers=headers, timeout=30)
    r.raise_for_status()
    data = r.json()
    klines = data["data"]["klines"]

    rows = []
    for line in klines:
        parts = line.split(",")
        # f51=日期, f52=开, f53=收, f54=高, f55=低, f56=量(股), f57=额(元), ...
        rows.append({
            "Date": pd.to_datetime(parts[0]),
            "Turnover_em_ak": float(parts[6]) / 1e8,  # 元 → 亿港元
        })
    df = pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)
    print(f"   东财·akshare: {df['Date'].min().date()}→{df['Date'].max().date()}  "
          f"共{len(df)}天  最新¥{df['Turnover_em_ak'].iloc[-1]:.0f}亿")
    return df


def fetch_hk_market_turnover(start_date: str, end_date: str) -> pd.DataFrame:
    """获取恒生指数日成交额（双源合并）→ 缓存

    新浪(akshare) + 东财(akshare)两个接口同时拉取，合并写入同一 CSV：
      Date, Turnover_sina, Turnover_em_ak
    缓存命中时直接返回。
    """
    os.makedirs("../data_cache", exist_ok=True)

    # — 缓存命中 —
    cached = _load_cache()
    if cached is not None and "Turnover_sina" in cached.columns and "Turnover_em_ak" in cached.columns:
        cs = cached["Date"].min().strftime("%Y%m%d")
        ce = cached["Date"].max().strftime("%Y%m%d")
        if cs <= start_date and ce >= end_date:
            r = cached[(cached["Date"] >= start_date) & (cached["Date"] <= end_date)]
            print(f"📦 [缓存命中] {start_date}~{end_date} → {len(r)} 条")
            return r.reset_index(drop=True)

    # — 双源拉取 —
    df_sina = fetch_hk_turnover_sina(start_date, end_date)

    df_em_ak = None
    try:
        df_em_ak = fetch_hk_turnover_em_ak(start_date, end_date)
    except Exception as e:
        print(f"   ⚠️ 东财(akshare)接口拉取失败，仅保留新浪数据: {e}")

    # 按日期合并
    if df_em_ak is not None:
        result = pd.merge(df_sina, df_em_ak, on="Date", how="outer").sort_values("Date")
    else:
        result = df_sina.copy()
        result["Turnover_em_ak"] = np.nan
    result.to_csv(CACHE_PATH, index=False)

    sina_ok = result["Turnover_sina"].notna().sum()
    em_ok = result["Turnover_em_ak"].notna().sum()
    print(f"\n💾 双源缓存: {result['Date'].min().date()}→{result['Date'].max().date()}  "
          f"共{len(result)}天")
    print(f"   新浪有效: {sina_ok}天  "
          f"东财·akshare有效: {em_ok}天")
    if sina_ok and em_ok:
        both = result.dropna(subset=["Turnover_sina", "Turnover_em_ak"])
        diff = (both["Turnover_sina"] - both["Turnover_em_ak"]).abs()
        print(f"   重叠 {len(both)} 天  |  平均差异: ¥{diff.mean():.0f}亿  "
              f"最大差异: ¥{diff.max():.0f}亿  "
              f"差异率均值: {((diff / both['Turnover_em_ak']) * 100).mean():.2f}%")

    r = result[(result["Date"] >= start_date) & (result["Date"] <= end_date)]
    return r.reset_index(drop=True)


def get_stock_turnover(ts_code: str, name: str,
                       start_date: str, end_date: str) -> pd.DataFrame:
    """个股成交额：直接从缓存读取"""
    from core.cache import load as cache_load
    df = cache_load(ts_code, "HK")
    if df is None or df.empty:
        print(f"⚠️ {name}: 无缓存")
        return pd.DataFrame()
    df = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)].copy()
    df["Turnover"] = df["Amount"] * 1000 / 1e8  # 千元→亿元
    return df[["Date", "Turnover"]].sort_values("Date").reset_index(drop=True)


def plot(market: pd.DataFrame, stocks: dict, path: str = "hk_market_turnover.html"):
    """交互式图表：双源恒生指数成交额 + 个股折线，可拖动缩放"""
    import plotly.graph_objects as go

    colors = {"小米": "#ff6b35", "美团": "#f5c842", "理想": "#2c3e50",
              "老铺": "#c0392b", "泡泡玛特": "#8e44ad"}

    fig = go.Figure()

    # — 恒生指数成交额（双源对比） —
    if not market.empty:
        # 新浪
        # if "Turnover_sina" in market.columns and market["Turnover_sina"].notna().any():
        #     fig.add_trace(go.Scatter(
        #         x=market["Date"], y=market["Turnover_sina"],
        #         mode="lines",
        #         name="恒生指数·新浪",
        #         line=dict(color="#e74c3c", width=0.8, dash="dash"),
        #         hovertemplate="新浪<br>%{x|%Y-%m-%d}<br>¥%{y:.0f}亿<extra></extra>",
        #     ))
        # 东财
        if "Turnover_em_ak" in market.columns and market["Turnover_em_ak"].notna().any():
            fig.add_trace(go.Scatter(
                x=market["Date"], y=market["Turnover_em_ak"],
                mode="lines",
                name="恒生指数·东财",
                line=dict(color="#2980b9", width=0.8),
                hovertemplate="东财<br>%{x|%Y-%m-%d}<br>¥%{y:.0f}亿<extra></extra>",
            ))

    # — 个股折线 —
    for name, df in stocks.items():
        if df.empty:
            continue
        c = colors.get(name, "#666")
        fig.add_trace(go.Scatter(
            x=df["Date"], y=df["Turnover"],
            mode="lines",
            name=name,
            line=dict(color=c, width=1.0),
            hovertemplate=f"{name}<br>%{{x|%Y-%m-%d}}<br>¥%{{y:.1f}}亿<extra></extra>",
        ))

    # 布局：像 Choice — 底部有范围滑块，支持拖拽缩放
    fig.update_layout(
        title=dict(
            text="港股日成交额 — 恒生指数(新浪vs东财) + 小米·美团·理想·老铺·泡泡玛特  (2018.01 ~ 2026.06)",
            font=dict(size=16),
        ),
        xaxis=dict(
            title="",
            type="date",
            rangeslider=dict(visible=True, thickness=0.05),  # 底部可拖动的范围滑块
            rangeselector=dict(  # 快捷时间按钮
                buttons=list([
                    dict(count=1, label="1月", step="month", stepmode="backward"),
                    dict(count=3, label="3月", step="month", stepmode="backward"),
                    dict(count=6, label="6月", step="month", stepmode="backward"),
                    dict(count=1, label="1年", step="year", stepmode="backward"),
                    dict(count=3, label="3年", step="year", stepmode="backward"),
                    dict(step="all", label="全部"),
                ])
            ),
        ),
        yaxis=dict(
            title="日成交额 (亿元 HKD)",
            title_font=dict(size=12),
            fixedrange=False,
        ),
        hovermode="x unified",
        legend=dict(
            orientation="h", y=1.12, x=0.5, xanchor="center",
            font=dict(size=11),
        ),
        template="plotly_white",
        height=650,
        margin=dict(l=60, r=40, t=80, b=40),
    )

    fig.write_html(path)
    print(f"📊 交互图表已保存: {path}")


if __name__ == "__main__":
    START, END = "20180101", "20260624"

    # 1. 市场（双源对比）
    print("1️⃣ 恒生指数日成交额（东财）...")
    mkt = fetch_hk_market_turnover(START, END)
    print(f"   日期范围: {mkt['Date'].min().date()} → {mkt['Date'].max().date()}")
    # if "Turnover_sina" in mkt.columns:
    #     s = mkt["Turnover_sina"].dropna()
    #     print(f"   新浪 — 均值¥{s.mean():.0f}亿  最新¥{s.iloc[-1]:.0f}亿")
    if "Turnover_em_ak" in mkt.columns:
        e = mkt["Turnover_em_ak"].dropna()
        print(f"   东财 — 均值¥{e.mean():.0f}亿  最新¥{e.iloc[-1]:.0f}亿")

    # 2. 个股
    print("\n2️⃣ 个股成交额...")
    stocks = {}
    for code, name in [("01810.HK","小米"), ("03690.HK","美团"), ("02015.HK","理想"),
                       ("06181.HK","老铺"), ("09992.HK","泡泡玛特")]:
        df = get_stock_turnover(code, name, START, END)
        if not df.empty:
            stocks[name] = df
            print(f"   {name}: {df['Date'].min().date()}→{df['Date'].max().date()}  "
                  f"均值¥{df['Turnover'].mean():.1f}亿  最大¥{df['Turnover'].max():.1f}亿")

    # 3. 绘图
    print("\n3️⃣ 生成交互图表...")
    plot(mkt, stocks, "hk_market_turnover.html")
    print("✅ 完成，用浏览器打开 hk_market_turnover.html")
