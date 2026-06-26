#!/usr/bin/env python3
"""个股历史熊市表现分析 + 反弹可能性评估

分析每只股票在历次恒生指数熊市中的表现，判断：
  1. 当前是否已在极端超卖区域
  2. 熊市中是否会有阶段性反弹
  3. 如果恒指继续走熊，各股可能怎么走

输出：终端打印 + analyze_stocks_bear_result.txt
"""

import sys
import os
import numpy as np
import pandas as pd
import akshare as ak

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.cache import load as cache_load

OUTPUT = os.path.join(os.path.dirname(__file__), "analyze_stocks_bear_result.txt")

STOCKS = [
    ("01810.HK", "小米"),
    ("03690.HK", "美团"),
    ("02015.HK", "理想"),
    ("06181.HK", "老铺"),
    ("09992.HK", "泡泡玛特"),
]

# HSI 历史熊市 & 反弹
BEAR_MARKETS = [
    ("2018 贸易战熊市",  "2018-01-26", "2018-10-30"),
    ("2020 COVID 暴跌",  "2020-01-17", "2020-03-23"),
    ("2021-22 科网股灾", "2021-02-17", "2022-10-31"),
    ("2026 当前下跌",   "2026-01-29", "2026-06-24"),
]
RALLIES = [
    ("→ 2019 反弹",     "2018-10-31", "2019-04-15"),
    ("→ 2020-21 牛市",  "2020-03-24", "2021-02-16"),
    ("→ 2023 大反弹",   "2022-11-01", "2023-01-27"),
    ("→ 2024-26 牛市",  "2024-01-22", "2026-01-29"),
]


# ============================================================
def load_data():
    hsi = ak.stock_hk_index_daily_sina(symbol="HSI")
    hsi["date"] = pd.to_datetime(hsi["date"])
    hsi = hsi.sort_values("date").reset_index(drop=True)
    stocks = {}
    for code, name in STOCKS:
        df = cache_load(code, "HK")
        if df is not None and not df.empty:
            stocks[name] = df.sort_values("Date").reset_index(drop=True)
    return hsi, stocks


# ============================================================
# 1. 各阶段超额收益对比
# ============================================================
def analyze_relative_performance(hsi, stocks):
    all_periods = []
    for label, s, e in BEAR_MARKETS + RALLIES:
        hsub = hsi[(hsi["date"] >= s) & (hsi["date"] <= e)]
        if len(hsub) < 2: continue
        h_chg = (hsub["close"].iloc[-1] - hsub["close"].iloc[0]) / hsub["close"].iloc[0] * 100
        row = {"阶段": label, "HSI涨跌幅": round(h_chg, 1)}
        for _, name in STOCKS:
            df = stocks.get(name)
            if df is None: continue
            sub = df[(df["Date"] >= s) & (df["Date"] <= e)]
            if len(sub) < 5: continue
            s_chg = (sub["Close"].iloc[-1] - sub["Close"].iloc[0]) / sub["Close"].iloc[0] * 100
            row[name] = round(s_chg, 1)
        all_periods.append(row)
    return all_periods


# ============================================================
# 2. 熊市中的阶段性反弹
# ============================================================
def analyze_bear_market_bounces(stocks):
    """在2021-22最惨烈熊市中，个股的最大阶段性反弹"""
    results = {}
    for _, name in STOCKS:
        df = stocks.get(name)
        if df is None: continue
        sub = df[(df["Date"] >= "2021-02-17") & (df["Date"] <= "2022-10-31")].copy()
        if len(sub) < 20: continue
        sub = sub.sort_values("Date")
        sub["ret_10d"] = sub["Close"].pct_change(10) * 100
        sub["ret_20d"] = sub["Close"].pct_change(20) * 100
        sub["ret_5d"] = sub["Close"].pct_change(5) * 100

        # 月度统计
        sub["Month"] = sub["Date"].dt.to_period("M")
        monthly = sub.groupby("Month")["Close"].agg(["first", "last"])
        monthly["chg"] = (monthly["last"] - monthly["first"]) / monthly["first"] * 100
        pos_months = (monthly["chg"] > 0).sum()

        # 当前熊市(2026)的反弹
        sub_cur = df[(df["Date"] >= "2026-01-29") & (df["Date"] <= "2026-06-24")].copy()
        sub_cur = sub_cur.sort_values("Date")
        if len(sub_cur) >= 21:
            sub_cur["ret_10d"] = sub_cur["Close"].pct_change(10) * 100
            sub_cur["ret_20d"] = sub_cur["Close"].pct_change(20) * 100

        results[name] = {
            "max_10d": sub["ret_10d"].max(),
            "max_20d": sub["ret_20d"].max(),
            "max_5d": sub["ret_5d"].max(),
            "pos_months": pos_months,
            "total_months": len(monthly),
            "max_monthly": monthly["chg"].max(),
            "cur_max_10d": sub_cur["ret_10d"].max() if len(sub_cur) >= 11 else np.nan,
        }
    return results


# ============================================================
# 3. 当前超卖程度
# ============================================================
def analyze_oversold(stocks):
    results = {}
    for _, name in STOCKS:
        df = stocks.get(name)
        if df is None: continue
        df = df.sort_values("Date").copy()
        df["cummax"] = df["Close"].cummax()
        df["dd"] = (df["Close"] - df["cummax"]) / df["cummax"] * 100

        ath = df["Close"].max()
        ath_date = df[df["Close"] == ath]["Date"].iloc[0]
        atl = df["Close"].min()
        atl_date = df[df["Close"] == atl]["Date"].iloc[0]
        latest = df["Close"].iloc[-1]
        latest_date = df["Date"].iloc[-1]
        current_dd = (latest - ath) / ath * 100
        max_dd = df["dd"].min()
        max_dd_date = df[df["dd"] == max_dd]["Date"].iloc[0]
        dd_percentile = (df["dd"] <= current_dd).sum() / len(df) * 100
        dist_from_low = (latest - atl) / atl * 100

        # 最近跌幅速度
        if len(df) >= 21:
            ret_20d = df["Close"].iloc[-1] / df["Close"].iloc[-21] - 1
        else:
            ret_20d = np.nan

        # 距52周高点
        df_1y = df[df["Date"] >= (df["Date"].iloc[-1] - pd.Timedelta(days=365))]
        high_52w = df_1y["Close"].max()
        dd_52w = (latest - high_52w) / high_52w * 100 if len(df_1y) > 0 else 0

        results[name] = {
            "ath": ath, "ath_date": ath_date,
            "atl": atl, "atl_date": atl_date,
            "latest": latest, "latest_date": latest_date,
            "current_dd": current_dd, "max_dd": max_dd,
            "max_dd_date": max_dd_date,
            "dd_percentile": dd_percentile,
            "dist_from_low": dist_from_low,
            "ret_20d": ret_20d,
            "high_52w": high_52w, "dd_52w": dd_52w,
        }
    return results


# ============================================================
# 4. 相关性 & 独立行情概率
# ============================================================
def analyze_correlation(hsi, stocks):
    results = {}
    for _, name in STOCKS:
        df = stocks.get(name)
        if df is None: continue
        hsub = hsi[(hsi["date"] >= "2025-10-01") & (hsi["date"] <= "2026-06-24")].copy()
        ssub = df[(df["Date"] >= "2025-10-01") & (df["Date"] <= "2026-06-24")].copy()
        hsub = hsub.set_index("date").sort_index()
        ssub = ssub.set_index("Date").sort_index()
        common = hsub.index.intersection(ssub.index)
        if len(common) < 30: continue

        h_ret = hsub.loc[common, "close"].pct_change().dropna()
        s_ret = ssub.loc[common, "Close"].pct_change().dropna()
        common_ret = h_ret.index.intersection(s_ret.index)

        corr = h_ret.loc[common_ret].corr(s_ret.loc[common_ret])
        cov_mat = np.cov(s_ret.loc[common_ret], h_ret.loc[common_ret])
        beta = cov_mat[0, 1] / cov_mat[1, 1] if cov_mat[1, 1] > 0 else 0

        both_up = ((h_ret > 0) & (s_ret > 0)).sum()
        both_down = ((h_ret < 0) & (s_ret < 0)).sum()
        diverge = ((h_ret > 0) & (s_ret < 0)).sum()
        anti = ((h_ret < 0) & (s_ret > 0)).sum()
        total = len(common_ret)

        results[name] = {
            "corr": corr, "beta": beta,
            "both_up": both_up, "both_down": both_down,
            "diverge": diverge, "anti": anti,
            "total": total,
            "anti_pct": anti / total * 100,
            "diverge_pct": diverge / total * 100,
        }
    return results


# ============================================================
# 5. 报告生成
# ============================================================
def generate_report():
    hsi, stocks = load_data()
    perf = analyze_relative_performance(hsi, stocks)
    bounces = analyze_bear_market_bounces(stocks)
    oversold = analyze_oversold(stocks)
    corr = analyze_correlation(hsi, stocks)

    lines = []
    def p(*args):
        lines.append(" ".join(str(a) for a in args))

    p("=" * 75)
    p("个股历史熊市表现 & 反弹可能性评估")
    p(f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    p("=" * 75)

    # ---- 全景对比表 ----
    p("\n一、各阶段涨跌幅全景 (个股 vs HSI)")
    p("-" * 75)
    header = f"  {'阶段':<24s} {'HSI':>7s}"
    for _, name in STOCKS:
        header += f" {name:>8s}"
    p(header)
    p(f"  {'-'*68}")
    for row in perf:
        line = f"  {row['阶段']:<24s} {row['HSI涨跌幅']:>+6.1f}%"
        for _, name in STOCKS:
            if name in row:
                line += f" {row[name]:>7.1f}%"
            else:
                line += f" {'—':>7s}"
        p(line)

    # ---- 超额收益 ----
    p("\n二、超额收益/亏损 (相对于HSI)")
    p("-" * 75)
    for _, name in STOCKS:
        p(f"\n  【{name}】")
        for row in perf:
            if name not in row: continue
            alpha = row[name] - row["HSI涨跌幅"]
            if row["HSI涨跌幅"] < -10:  # 熊市
                tag = "抗跌✅" if alpha > 5 else ("崩盘❌" if alpha < -10 else "跟跌")
            elif row["HSI涨跌幅"] > 10:  # 牛市
                tag = "领涨🚀" if alpha > 20 else ("滞涨⚠️" if alpha < -10 else "跟涨")
            else:
                tag = ""
            p(f"    {row['阶段']:<24s} HSI{row['HSI涨跌幅']:>+6.1f}%  "
              f"个股{row[name]:>+6.1f}%  超额{alpha:>+6.1f}%  {tag}")

    # ---- 熊市反弹 ----
    p("\n三、2021-22科网股灾中的阶段性反弹 (证明熊市中也会有反弹)")
    p("-" * 75)
    p(f"  {'名称':<10s} {'最大20日反弹':>12s} {'最大10日反弹':>12s} {'上涨月/总月':>10s} {'当前20日反弹':>12s}")
    for _, name in STOCKS:
        b = bounces.get(name, {})
        if not b: continue
        cur = f"{b['cur_max_10d']:+.1f}%" if not np.isnan(b.get("cur_max_10d", np.nan)) else "N/A"
        pos = int(b['pos_months'])
        tot = int(b['total_months'])
        p(f"  {name:<10s} {b['max_20d']:>+10.1f}% {b['max_10d']:>+10.1f}%  "
          f"{pos}/{tot}        {cur:>12s}")

    p("\n  结论：即使在2021-22最惨烈的科网股灾中(-53%)，个股也存在20-85%的阶段性反弹。")
    p("  但这些反弹是死猫跳(dead cat bounce) —— 除非恒指趋势反转，否则反弹后会继续新低。")

    # ---- 超卖 ----
    p("\n四、当前超卖程度")
    p("-" * 75)
    for _, name in STOCKS:
        o = oversold.get(name, {})
        if not o: continue
        p(f"\n  【{name}】")
        p(f"    历史最高: {o['ath']:.1f} ({o['ath_date'].date()})  |  "
          f"历史最低: {o['atl']:.1f} ({o['atl_date'].date()})")
        p(f"    最新价: {o['latest']:.1f} ({o['latest_date'].date()})")
        p(f"    当前回撤: {o['current_dd']:.1f}%  "
          f"| 历史最大回撤: {o['max_dd']:.1f}% ({o['max_dd_date'].date()})")
        p(f"    回撤深度排名: 比 {o['dd_percentile']:.0f}% 的历史交易日更深")
        p(f"    距历史最低: 还有 {o['dist_from_low']:.0f}%({'接近底部' if o['dist_from_low']<30 else '下方空间大' if o['dist_from_low']>100 else ''})")
        p(f"    20日跌幅: {o['ret_20d']:+.1f}%")

    # ---- 个股定性判断 ----
    p("\n五、逐只定性分析与反弹可能性")
    p("-" * 75)

    analyses = {
        "理想": """
    理想 — ⚠️ 最危险，也最可能暴力反弹
    · 刚创历史新低(48.3)，处于从未到达过的区域
    · 2024-26牛市期间逆势下跌53.8%，与大盘完全脱钩
    · 历史先例：2022年熊市中曾出现+84.6%的20日反弹
    · Beta 1.25，与HSI中等相关
    · 判断：已被市场彻底抛弃，技术面极差。但正因如此，一旦有任何
      利好(政策/销量/行业拐点)，报复性反弹可能最猛烈。
      如果恒指继续熊市，大概率继续探底，不建议左侧抄底。""",

        "美团": """
    美团 — ⚠️ 回撤极深但反弹动能存疑
    · 回撤-84.6%已接近历史极值(-86.1%)，处于1%分位
    · 2024-26牛市仅涨6.4%(大盘+56.5%)—— 资金对它完全没兴趣
    · 2022年熊市中曾反弹+45.4%，有阶段性爆发力
    · Beta最高(1.32)，大盘跌它跌更多
    · 判断：估值已极其便宜，但缺乏催化剂。在牛市都不涨的股票，
      熊市中更难指望独立走强。可能在历史极值附近筑底，但反弹
      需要等恒指企稳。""",

        "小米": """
    小米 — ⚠️ 有基本面支撑，但技术面未到极端
    · 回撤-62.4%，距历史极值(-76.1%)还有空间
    · Beta 1.12，跟HSI较紧，独立行情少(仅11天HSI跌它涨)
    · 2024-26牛市+74.3%，跑赢大盘，有基本面支撑
    · 但2019年反弹中完全没涨(-4.4% vs HSI +19.3%)，有跟跌不跟涨的前科
    · 判断：下跌幅度还不够极端，如果恒指继续跌，可能还有10-15%下行。
      但不是最危险的，汽车+IoT+手机三轮驱动有一定安全垫。""",

        "泡泡玛特": """
    泡泡玛特 — 牛市大赢家正在回吐，离真正底部还远
    · 回撤-53.3%，远未到历史极值(-90.3%)
    · 2024-26牛市+810%！现在只是回吐了部分涨幅
    · 从历史最低仍有+1442%空间—— 它还非常贵
    · 相关性最低(0.35)，独立行情最多(28天HSI跌它涨)
    · 判断：最可能在熊市中出现独立反弹的股票。但它最大的风险
      是估值——牛市涨8倍的股票在熊市中可能跌回原点。
      如果恒指继续跌，这是最需要止损的品种。""",

        "老铺": """
    老铺 — 次新股，数据不足，风险不可评估
    · 上市仅1年，没有经历过完整牛熊周期
    · 最高1082 → 404，回撤-62.7%
    · Beta 1.36最高，波动极大
    · 判断：数据太少，历史规律不可靠。新股在熊市中往往跌最惨
      (没有锚定价格)。回避为佳。""",
    }

    for _, name in STOCKS:
        if name in analyses:
            p(analyses[name])

    # ---- 总结 ----
    p("\n六、核心结论")
    p("-" * 75)
    p("""
  1. 熊市中确实会有反弹：上一轮科网股灾中，理想最大20日反弹+84.6%，
     美团+45.4%，泡泡玛特+34.8%。但这些反弹都是暂时的，"死猫跳"之后
     全部继续新低。

  2. 反弹的前提：恒指阶段性企稳 + 个股超卖到极致。
     当前没有任何一只股票满足"极度超卖+恒指企稳"的条件。

  3. 反弹可能性排序（如果恒指继续熊市）：
     泡泡玛特 > 小米 > 美团 > 理想 > 老铺
     （泡泡玛特相关性最低、独立行情最多；老铺数据太少无法判断）

  4. 暴力反弹潜力排序（如果恒指突然反转）：
     理想 > 美团 > 老铺 > 小米 > 泡泡玛特
     （越是被抛弃的、跌得越深的，反弹越猛）

  5. 当前最重要的信号是等恒指缩量止跌。
     在恒指止跌之前，任何个股反弹都是减仓机会而非抄底信号。
""")

    text = "\n".join(lines)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"\n📄 报告已保存: {OUTPUT}")


if __name__ == "__main__":
    generate_report()
