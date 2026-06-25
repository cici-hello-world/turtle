#!/usr/bin/env python3
"""A股/港股 2026年以来每日总成交额统计 & 对比图

数据来源：
  - A股：Tushare index_daily（000001.SH 上证综指 + 399001.SZ 深证成指）
  - 港股：AKShare stock_hk_index_daily_sina（恒生指数 HSI）
"""

import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np
import tushare as ts
import akshare as ak
from tushare.util.dateu import today


# ============================================================
# 1. 获取 A 股每日总成交额（上证 + 深证）
# ============================================================
def fetch_a_share_turnover() -> pd.DataFrame:
    """返回 DataFrame，含 trade_date, amount_a (亿元)"""
    token = os.environ.get("TUSHARE_TOKEN")
    if not token:
        raise RuntimeError("请先设置 TUSHARE_TOKEN 环境变量")
    pro = ts.pro_api(token)

    # 分段拉取 SSE & SZSE 指数日线
    frames_sh, frames_sz = [], []
    for ts_code, storage in [("000001.SH", frames_sh), ("399107.SZ", frames_sz)]:
        # 按月分段，避免单次拉取过多
        for year in range(2025, 2026):
            for month in range(1, 12):
                start = f"{year}{month:02d}01"
                end = f"{year}{month:02d}31"
                df = pro.index_daily(ts_code=ts_code, start_date=start, end_date=end)
                if df.empty:
                    break
                storage.append(df)
    df_sh = pd.concat(frames_sh, ignore_index=True)
    df_sz = pd.concat(frames_sz, ignore_index=True)

    # 合并：按日期对齐，amount 相加（单位：千元 → 亿元）
    df_sh = df_sh[["trade_date", "amount"]].rename(columns={"amount": "amount_sh"})
    df_sz = df_sz[["trade_date", "amount"]].rename(columns={"amount": "amount_sz"})
    df = pd.merge(df_sh, df_sz, on="trade_date", how="inner")
    # tushare index_daily amount 单位：千元；转为亿元 (1亿 = 10^8)
    df["amount_a"] = (df["amount_sh"] + df["amount_sz"]) * 1000 / 1e8
    df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
    df = df.sort_values("trade_date").reset_index(drop=True)
    print(f"✅ A股：{len(df)} 个交易日，日均成交额 {df['amount_a'].mean():.0f} 亿元")
    return df[["trade_date", "amount_a"]]


# ============================================================
# 2. 获取港股每日总成交额（恒生指数成分股 turnover）
# ============================================================
def fetch_hk_turnover() -> pd.DataFrame:
    """返回 DataFrame，含 date, amount_hk (亿港元)"""
    df = ak.stock_hk_index_daily_sina(symbol="HSI")
    # amount 单位：港元；转为亿港元
    df = df[["date", "amount"]].copy()
    df["amount_hk"] = df["amount"] / 1e8
    df["date"] = pd.to_datetime(df["date"])
    # 只保留 2026 年
    df = df[df["date"]>="2025-01-01"].copy()
    df = df.sort_values("date").reset_index(drop=True)
    print(f"✅ 港股：{len(df)} 个交易日，日均成交额 {df['amount_hk'].mean():.0f} 亿港元")
    return df[["date", "amount_hk"]]


# ============================================================
# 3. 合并并绘图
# ============================================================
def plot_turnover(df_a: pd.DataFrame, df_hk: pd.DataFrame):
    """绘制双市场每日成交额折线图"""
    # 对齐日期
    df_a = df_a.rename(columns={"trade_date": "date"})
    merged = pd.merge(df_a, df_hk, on="date", how="outer").sort_values("date")

    fig, ax1 = plt.subplots(figsize=(18, 7))

    # A股 —— 左轴
    color_a = "#e74c3c"
    ax1.plot(merged["date"], merged["amount_a"],
             color=color_a, linewidth=1.2, marker=".", markersize=2, label="A 股（亿元 RMB）")
    ax1.set_ylabel("A 股成交额（亿元 RMB）", color=color_a, fontsize=11)
    ax1.tick_params(axis="y", labelcolor=color_a)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    # 港股 —— 右轴
    ax2 = ax1.twinx()
    color_hk = "#2e86c1"
    ax2.plot(merged["date"], merged["amount_hk"],
             color=color_hk, linewidth=1.2, marker=".", markersize=2, label="港股（亿港元）")
    ax2.set_ylabel("港股成交额（亿港元）", color=color_hk, fontsize=11)
    ax2.tick_params(axis="y", labelcolor=color_hk)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    # X 轴日期格式
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=9)

    # 标题 & 网格
    ax1.set_title("2026 年 A 股 / 港股每日总成交额对比", fontsize=15, fontweight="bold", pad=18)
    ax1.grid(True, alpha=0.3, linestyle="--")

    # 图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=10)

    # 标注日均线（30 日滚动均值）
    merged["a_ma30"] = merged["amount_a"].rolling(30, min_periods=1).mean()
    merged["hk_ma30"] = merged["amount_hk"].rolling(30, min_periods=1).mean()
    ax1.plot(merged["date"], merged["a_ma30"], color="#c0392b", linewidth=1.8,
             linestyle="--", alpha=0.7, label="A 股 30日均值")
    ax2.plot(merged["date"], merged["hk_ma30"], color="#1a5276", linewidth=1.8,
             linestyle="--", alpha=0.7, label="港股 30日均值")

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), f"market_daily_turnover_{merged["date"][0].year}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n📈 图表已保存至：{out_path}")
    plt.show()


# ============================================================
# 4. 统计摘要
# ============================================================
def print_summary(df_a: pd.DataFrame, df_hk: pd.DataFrame):
    """打印关键统计数据"""
    print("\n" + "=" * 65)
    print("📊 2026 年以来每日总成交额统计摘要")
    print("=" * 65)

    for label, df, col, unit in [
        ("A 股", df_a, "amount_a", "亿元 RMB"),
        ("港股", df_hk, "amount_hk", "亿港元"),
    ]:
        s = df[col]
        print(f"\n  【{label}】({len(df)} 个交易日)")
        print(f"    均值：  {s.mean():>10,.0f} {unit}")
        print(f"    中位数：{s.median():>10,.0f} {unit}")
        print(f"    最高：  {s.max():>10,.0f} {unit}  ({df.loc[s.idxmax(), df.columns[0]].strftime('%m/%d')})")
        print(f"    最低：  {s.min():>10,.0f} {unit}  ({df.loc[s.idxmin(), df.columns[0]].strftime('%m/%d')})")
        print(f"    最新：  {s.iloc[-1]:>10,.0f} {unit}  ({df.iloc[-1, 0].strftime('%m/%d')})")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    # 设置中文字体（macOS）
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "PingFang SC"]
    plt.rcParams["axes.unicode_minus"] = False

    print("⏳ 正在获取 A 股成交额数据...")
    df_a = fetch_a_share_turnover()

    print("⏳ 正在获取港股成交额数据...")
    df_hk = fetch_hk_turnover()

    print_summary(df_a, df_hk)
    plot_turnover(df_a, df_hk)
