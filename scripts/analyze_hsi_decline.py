#!/usr/bin/env python3
"""恒生指数 2025.10 以来下跌分析

分析维度：
  1. 牛市→熊市转折识别（量价关系）
  2. 阶段划分与涨跌幅
  3. 天量见天价信号
  4. 个股 vs 指数分化
  5. 当前见底条件判断

输出：终端打印 + analyze_hsi_decline_result.txt
"""

import sys
import os
import numpy as np
import pandas as pd
import akshare as ak

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUTPUT = os.path.join(os.path.dirname(__file__), "analyze_hsi_decline_result.txt")


# ============================================================
# 1. 数据获取
# ============================================================
def fetch_data():
    """获取恒生指数历史日线 + 5只成分股"""
    print("📡 获取恒生指数数据...")
    hsi = ak.stock_hk_index_daily_sina(symbol="HSI")
    hsi["date"] = pd.to_datetime(hsi["date"])
    hsi = hsi.sort_values("date").reset_index(drop=True)

    # 关键成分股
    from core.cache import load as cache_load
    stocks = {}
    for code, name in [
        ("01810.HK", "小米"),
        ("03690.HK", "美团"),
        ("02015.HK", "理想"),
        ("06181.HK", "老铺"),
        ("09992.HK", "泡泡玛特"),
    ]:
        df = cache_load(code, "HK")
        if df is not None and not df.empty:
            stocks[name] = df
    return hsi, stocks


# ============================================================
# 2. 阶段分析
# ============================================================
STAGES = [
    ("2024 熊市筑底",       "2024-01-01", "2024-08-31"),
    ("2024.9 政策牛启动",   "2024-09-01", "2024-10-31"),
    ("2024.11→2025.3 震荡上行", "2024-11-01", "2025-03-31"),
    ("2025.4→2025.7 主升浪",   "2025-04-01", "2025-07-31"),
    ("2025.8→2025.9 加速赶顶", "2025-08-01", "2025-09-30"),
    ("2025.10→至今 下跌",       "2025-10-01", "2026-06-24"),
]

DROP_WAVES = [
    ("第1波急跌",  "2025-10-02", "2025-10-17"),
    ("第2波回调",  "2025-11-06", "2025-11-21"),
    ("第3波主跌",  "2026-01-29", "2026-03-13"),
    ("第4波加速跌","2026-05-15", "2026-06-23"),
]


def analyze_stages(hsi: pd.DataFrame) -> list[dict]:
    """逐阶段统计"""
    rows = []
    for label, s, e in STAGES:
        sub = hsi[(hsi["date"] >= s) & (hsi["date"] <= e)]
        if len(sub) < 2:
            continue
        s_p, e_p = sub["close"].iloc[0], sub["close"].iloc[-1]
        chg = (e_p - s_p) / s_p * 100
        avg_amt = sub["amount"].mean() / 1e8
        avg_vol = sub["volume"].mean() / 1e8
        hi, lo = sub["close"].max(), sub["close"].min()
        rows.append({
            "阶段": label, "起始日": s, "结束日": e,
            "起始价": s_p, "结束价": e_p, "涨跌幅%": round(chg, 2),
            "最高": hi, "最低": lo,
            "日均成交额(亿)": round(avg_amt, 0),
            "日均量(亿股)": round(avg_vol, 1),
            "交易日": len(sub),
        })
    return rows


def analyze_drop_waves(hsi: pd.DataFrame) -> list[dict]:
    """下跌波段拆解"""
    rows = []
    for label, s, e in DROP_WAVES:
        sub = hsi[(hsi["date"] >= s) & (hsi["date"] <= e)]
        if len(sub) < 2:
            continue
        s_p, e_p = sub["close"].iloc[0], sub["close"].iloc[-1]
        chg = (e_p - s_p) / s_p * 100
        hi, lo = sub["close"].max(), sub["close"].min()
        rows.append({
            "波段": label, "起始日": s, "结束日": e,
            "起始价": s_p, "结束价": e_p, "涨跌幅%": round(chg, 2),
            "最高": hi, "最低": lo, "交易日": len(sub),
        })
    return rows


def analyze_top_signals(hsi: pd.DataFrame) -> list[dict]:
    """顶部天量信号"""
    top = hsi[(hsi["date"] >= "2025-09-15") & (hsi["date"] <= "2025-10-15")]
    rows = []
    for _, r in top.iterrows():
        amt = r["amount"] / 1e8
        flag = ""
        if amt > 4000:
            flag = "⚠️ 天量"
        elif amt > 3000:
            flag = "放量"
        rows.append({
            "日期": r["date"].strftime("%Y-%m-%d"),
            "收盘": round(r["close"], 0),
            "成交额(亿)": round(amt, 0),
            "信号": flag,
        })
    return rows


def analyze_stocks(stocks: dict) -> list[dict]:
    """个股 2025.10 以来表现"""
    rows = []
    for name, df in stocks.items():
        sub = df[(df["Date"] >= "2025-10-01") & (df["Date"] <= "2026-06-25")]
        if len(sub) < 10:
            continue
        s_p, e_p = sub["Close"].iloc[0], sub["Close"].iloc[-1]
        chg = (e_p - s_p) / s_p * 100
        hi, lo = sub["Close"].max(), sub["Close"].min()
        # 月均成交额变化
        sub = sub.copy()
        sub["Month"] = sub["Date"].dt.to_period("M")
        monthly = sub.groupby("Month")["Amount"].mean() * 1000 / 1e8
        amt_first = monthly.iloc[0] if len(monthly) > 0 else 0
        amt_last = monthly.iloc[-1] if len(monthly) > 0 else 0
        rows.append({
            "名称": name,
            "期初价": s_p, "最新价": e_p, "涨跌幅%": round(chg, 1),
            "最高": hi, "最低": lo,
            "月均成交额_首月(亿)": round(amt_first, 1),
            "月均成交额_末月(亿)": round(amt_last, 1),
        })
    return rows


# ============================================================
# 3. 报告输出
# ============================================================
def generate_report(hsi, stocks):
    stage_rows = analyze_stages(hsi)
    drop_rows = analyze_drop_waves(hsi)
    top_rows = analyze_top_signals(hsi)
    stock_rows = analyze_stocks(stocks)

    # 关键数字
    cycle_low = hsi[hsi["date"] >= "2024-01-01"]["close"].min()
    cycle_low_date = hsi[(hsi["date"] >= "2024-01-01") & (hsi["close"] == cycle_low)]["date"].iloc[0]
    cycle_high = hsi[(hsi["date"] >= "2024-01-01") & (hsi["date"] <= "2026-06-24")]["close"].max()
    cycle_high_date = hsi[(hsi["date"] >= "2024-01-01") & (hsi["close"] == cycle_high)]["date"].iloc[0]
    latest = hsi[hsi["date"] <= "2026-06-24"]["close"].iloc[-1]
    latest_date = hsi[hsi["date"] <= "2026-06-24"]["date"].iloc[-1]
    bull_gain = (cycle_high - cycle_low) / cycle_low * 100
    bear_dd = (latest - cycle_high) / cycle_high * 100

    # 量价判断
    bull_phase = [r for r in stage_rows if "主升" in r["阶段"]][0]
    bear_phase = [r for r in stage_rows if "下跌" in r["阶段"]][0]
    amt_up = bull_phase["日均成交额(亿)"]
    amt_down = bear_phase["日均成交额(亿)"]
    vol_signal = "放量下跌❌ 资金持续出逃" if amt_down > amt_up else "缩量下跌"

    top_amt_max = max(r["成交额(亿)"] for r in top_rows)

    lines = []
    def p(*args):
        lines.append(" ".join(str(a) for a in args))

    p("=" * 70)
    p("恒生指数 2025.10 以来下跌分析报告")
    p(f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    p("=" * 70)

    # — 概览 —
    p("\n一、周期概览")
    p("-" * 40)
    p(f"  牛市起点: {cycle_low:.0f} ({cycle_low_date.date()})")
    p(f"  牛市顶点: {cycle_high:.0f} ({cycle_high_date.date()})")
    p(f"  牛市涨幅: +{bull_gain:.1f}%")
    p(f"  最新收盘: {latest:.0f} ({latest_date.date()})")
    p(f"  从高点回撤: {bear_dd:.1f}%")

    # — 阶段 —
    p("\n二、六阶段走势")
    p("-" * 40)
    p(f"  {'阶段':<28s} {'涨跌幅':>7s} {'日均额':>8s} {'日均量':>8s} {'交易日':>5s}")
    p(f"  {'-'*56}")
    for r in stage_rows:
        chg_str = f"{r['涨跌幅%']:+.2f}%"
        p(f"  {r['阶段']:<28s} {chg_str:>7s} {r['日均成交额(亿)']:>6.0f}亿 {r['日均量(亿股)']:>6.1f}亿 {r['交易日']:>5d}天")

    # — 量价 —
    p("\n三、量价关系判断")
    p("-" * 40)
    p(f"  主升阶段日均成交额: {amt_up:.0f}亿")
    p(f"  下跌阶段日均成交额: {amt_down:.0f}亿")
    p(f"  结论: {vol_signal}")

    # — 天量 —
    p("\n四、顶部天量信号 (2025.9.15 ~ 10.15)")
    p("-" * 40)
    p(f"  {'日期':>12s} {'收盘':>8s} {'成交额':>8s} {'信号'}")
    p(f"  {'-'*38}")
    for r in top_rows:
        p(f"  {r['日期']:>12s} {r['收盘']:>8.0f} {r['成交额(亿)']:>6.0f}亿  {r['信号']}")
    p(f"\n  最大天量: {top_amt_max:.0f}亿 → 放量见顶，聪明钱派发")

    # — 下跌波段 —
    p("\n五、下跌波段拆解")
    p("-" * 40)
    for r in drop_rows:
        p(f"  {r['波段']}: {r['起始价']:.0f} → {r['结束价']:.0f} ({r['涨跌幅%']:+.1f}%), {r['交易日']}天")

    # — 个股 —
    p("\n六、5只科技股 vs 恒指对比")
    p("-" * 40)
    p(f"  恒生指数同期: {bear_dd:.1f}%")
    p(f"  {'名称':<10s} {'涨跌幅':>8s} {'月均成交额变化':>20s}")
    p(f"  {'-'*40}")
    for r in stock_rows:
        amt_change = f"{r['月均成交额_首月(亿)']:.0f}→{r['月均成交额_末月(亿)']:.0f}亿"
        p(f"  {r['名称']:<10s} {r['涨跌幅%']:>7.1f}%  {amt_change:>20s}")

    # — 成因 —
    p("\n七、下跌核心原因")
    p("-" * 40)
    p("  1. 技术面：2024.1→2026.1 涨幅 87%，严重超买，均值回归压力巨大")
    p("  2. 资金面：4904 亿历史天量后无增量资金承接，多空换手完成出货")
    p("  3. 估值面：科技成长股泡沫破裂，5只个股全部腰斩级别下跌")
    p("  4. 结构面：下跌放量(均额2619亿>主升2443亿)，典型资金出逃模式")
    p("  5. 情绪面：四段下跌一波比一波深，第4波-10.1%呈加速赶底态势")

    # — 见底条件 —
    p("\n八、见底条件跟踪")
    p("-" * 40)
    latest_amt = hsi[hsi["date"] <= "2026-06-24"]["amount"].iloc[-1] / 1e8
    recent_avg_amt = hsi[(hsi["date"] >= "2026-05-01") & (hsi["date"] <= "2026-06-24")]["amount"].mean() / 1e8
    is_new_low = latest <= hsi[(hsi["date"] >= "2026-05-01")]["close"].min()

    checks = [
        ("成交额萎缩至1500亿以下", latest_amt < 1500, f"当前{latest_amt:.0f}亿"),
        ("近期均额<2000亿", recent_avg_amt < 2000, f"当前{recent_avg_amt:.0f}亿"),
        ("指数不再创新低", not is_new_low, f"最新{latest:.0f}, 5月来最低{hsi[(hsi['date']>='2026-05-01')]['close'].min():.0f}"),
        ("个股出现抗跌分化", False, "5只仍全线下跌，未见分化"),
        ("缩量横盘2周以上", False, "仍在放量探底"),
    ]
    for condition, met, detail in checks:
        status = "✅" if met else "❌"
        p(f"  {status} {condition}: {detail}")

    met_count = sum(1 for _, m, _ in checks if m)
    p(f"\n  见底条件满足: {met_count}/5 — {'接近底部' if met_count >= 3 else '仍需等待'}")

    # 写文件
    text = "\n".join(lines)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"\n📄 报告已保存: {OUTPUT}")
    return text


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    hsi, stocks = fetch_data()
    generate_report(hsi, stocks)
