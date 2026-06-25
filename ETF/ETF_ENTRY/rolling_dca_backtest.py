#!/usr/bin/env python3
"""滚动5年 DCA 定投回测 —— 步长1个月，统计各窗口 XIRR 分布
第一步：查询投资周期
用法：
  python backtest_entry/rolling_dca_backtest.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tushare as ts
from pandas import DataFrame

from core.config import BacktestConfig
from core.data import fetch_data
from ETF.dca_utils import find_investment_dates, find_daily_investment_dates, simulate_dca, _compute_xirr

# ============================================================
# 配置
# ============================================================
TS_CODE = "513500.SH"
STOCK_NAME = "标普500ETF"
MONTHLY_AMOUNT = 3000.0
DAILY_AMOUNT = 100.0
PER_SHARES = 100
TARGET_DAY = 9
WINDOW_YEARS = 10
STEP_MONTHS = 1       # 步长（月）

out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ETF_dca_rolling")
os.makedirs(out_dir, exist_ok=True)


# ============================================================
# 1. 获取 ETF 上市日期
# ============================================================
def get_etf_list_date(ts_code: str,sn:str) -> str:
    """查询 ETF 上市日期，返回 'YYYYMMDD'"""
    token = os.environ.get("TUSHARE_TOKEN")
    if not token:
        raise RuntimeError("❌ 未设置 TUSHARE_TOKEN")
    ts.set_token(token)
    pro = ts.pro_api()

    df = pro.fund_basic(ts_code=ts_code, market="E")
    if df.empty:
        raise ValueError(f"❌ 未找到 ETF {ts_code} 的基本信息")
    list_date = str(df.iloc[0]["list_date"])
    print(f"📌 {sn} ({ts_code}) 上市日期: {list_date}")
    return list_date


# ============================================================
# 2. 生成滚动窗口
# ============================================================
def generate_rolling_windows(start_date: str, end_date: str,
                             window_years: int, step_months: int) -> list:
    """生成滚动窗口 (start, end) 列表"""
    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date)
    windows = []

    current = start_dt
    while True:
        window_end = current + pd.DateOffset(years=window_years)
        if window_end > end_dt:
            break
        windows.append((current.strftime("%Y%m%d"), window_end.strftime("%Y%m%d")))
        current += pd.DateOffset(months=step_months)

    print(f"✅ 生成 {len(windows)} 个滚动窗口"
          f"（{windows[0][0]} → {windows[-1][1]}）")
    return windows


# ============================================================
# 3. 滚动回测
# ============================================================
def run_rolling_backtest(df_full: pd.DataFrame, windows: list) -> list:
    """对每个窗口执行 DCA 模拟并计算 XIRR"""
    results = []
    total = len(windows)

    for i, (start, end) in enumerate(windows):
        # 截取窗口数据
        mask = (df_full["Date"] >= start) & (df_full["Date"] <= end)
        df_window = df_full[mask].copy()

        if df_window.empty:
            results.append({
                "开始日期": pd.Timestamp(start),
                "结束日期": pd.Timestamp(end),
                "投资期数": 0,
                "总投入": 0,
                "最终市值": 0,
                "总收益率": np.nan,
                "XIRR": np.nan,
            })
            continue

        # 定位投资日
        invest_dates = find_investment_dates(df_window, TARGET_DAY)
        if len(invest_dates) < 12:  # 至少需要1年数据
            results.append({
                "开始日期": pd.Timestamp(start),
                "结束日期": pd.Timestamp(end),
                "投资期数": len(invest_dates),
                "总投入": 0,
                "最终市值": 0,
                "总收益率": np.nan,
                "XIRR": np.nan,
            })
            continue

        # DCA 模拟
        sim = simulate_dca(df_window, invest_dates, MONTHLY_AMOUNT, PER_SHARES)

        # XIRR
        inv_df = sim["investments"]
        cashflows = []
        cf_dates = []
        for _, inv in inv_df.iterrows():
            if inv["买入手数"] > 0:
                cashflows.append(-inv["买入金额"])
                cf_dates.append(inv["投资日期"])
        cashflows.append(sim["final_value"])
        cf_dates.append(pd.Timestamp(end))
        xirr = _compute_xirr(cashflows, cf_dates) if len(cashflows) > 1 else None

        results.append({
            "开始日期": pd.Timestamp(start),
            "结束日期": pd.Timestamp(end),
            "投资期数": len(invest_dates),
            "总投入": sim["total_invested"],
            "最终市值": sim["final_value"],
            "总收益率": (sim["final_value"] - sim["total_invested"]) / sim["total_invested"] * 100,
            "XIRR": xirr * 100 if xirr is not None else np.nan,
        })

        if (i + 1) % 50 == 0 or i == 0:
            xirr_str = f"{results[-1]['XIRR']:.2f}%" if not np.isnan(results[-1]["XIRR"]) else "N/A"
            print(f"  进度: {i+1}/{total}  —  {start[:6]} → {end[:6]}  XIRR: {xirr_str}")

    return results


# ============================================================
# 3b. 滚动回测 —— 每日定投版
# ============================================================
def run_rolling_backtest_daily(df_full: pd.DataFrame, windows: list) -> list:
    """对每个窗口执行每日 DCA 模拟并计算 XIRR"""
    results = []
    total = len(windows)

    for i, (start, end) in enumerate(windows):
        mask = (df_full["Date"] >= start) & (df_full["Date"] <= end)
        df_window = df_full[mask].copy()

        if df_window.empty:
            results.append({
                "开始日期": pd.Timestamp(start),
                "结束日期": pd.Timestamp(end),
                "投资期数": 0,
                "总投入": 0,
                "最终市值": 0,
                "总收益率": np.nan,
                "XIRR": np.nan,
            })
            continue

        # 每日都是投资日
        invest_dates = find_daily_investment_dates(df_window)
        if len(invest_dates) < 250:  # 至少需要1年交易日
            results.append({
                "开始日期": pd.Timestamp(start),
                "结束日期": pd.Timestamp(end),
                "投资期数": len(invest_dates),
                "总投入": 0,
                "最终市值": 0,
                "总收益率": np.nan,
                "XIRR": np.nan,
            })
            continue

        # DCA 模拟
        sim = simulate_dca(df_window, invest_dates, DAILY_AMOUNT, PER_SHARES)

        # XIRR
        inv_df = sim["investments"]
        cashflows = []
        cf_dates = []
        for _, inv in inv_df.iterrows():
            if inv["买入手数"] > 0:
                cashflows.append(-inv["买入金额"])
                cf_dates.append(inv["投资日期"])
        cashflows.append(sim["final_value"])
        cf_dates.append(pd.Timestamp(end))
        xirr = _compute_xirr(cashflows, cf_dates) if len(cashflows) > 1 else None

        results.append({
            "开始日期": pd.Timestamp(start),
            "结束日期": pd.Timestamp(end),
            "投资期数": len(invest_dates),
            "总投入": sim["total_invested"],
            "最终市值": sim["final_value"],
            "总收益率": (sim["final_value"] - sim["total_invested"]) / sim["total_invested"] * 100,
            "XIRR": xirr * 100 if xirr is not None else np.nan,
        })

        if (i + 1) % 50 == 0 or i == 0:
            xirr_str = f"{results[-1]['XIRR']:.2f}%" if not np.isnan(results[-1]["XIRR"]) else "N/A"
            print(f"  进度: {i+1}/{total}  —  {start[:6]} → {end[:6]}  XIRR: {xirr_str}")

    return results


# ============================================================
# 4. 统计（含复合评分选取）
# ============================================================
def compute_stats(results: list, strategy: str = "balanced") -> dict:
    """计算滚动 XIRR 分布统计 + 复合评分选取最有预测价值的窗口

    分布统计（描述整体）：XIRR中位数、算术均值、标准差、分位数、亏损占比 等

    复合评分选取：不简单取分位数，而是对每个窗口计算综合得分，
    同时考虑回报（XIRR）、稳定性（偏离中位数的程度）、下行风险（亏损占比）。

    每个窗口的评分系数会写回 results，随 Excel「滚动窗口明细」一起保存。

    策略：
      - conservative: 偏爱接近中位数且可靠的窗口，惩罚偏离和下行风险
      - balanced:     奖励高于中位数，惩罚下行偏离和极端上行异常值
      - aggressive:   偏爱高 XIRR，对偏离和下行容忍度高
    """
    df = pd.DataFrame(results).dropna(subset=["XIRR"])
    n = len(df)

    if n == 0:
        return {"样本总数": 0}

    # 按 XIRR 排序
    df_sorted = df.sort_values("XIRR").reset_index(drop=True)
    median_row = df_sorted.iloc[n // 2]

    xirrs = df["XIRR"].values
    loss_ratio = (xirrs < 0).sum() / n * 100
    median = median_row["XIRR"]
    std = max(np.std(xirrs, ddof=1), 0.01)
    safe_ratio = max(1 - loss_ratio / 100, 0.01)

    # ---- 分布统计 ----
    stats = {
        "样本总数": n,
        "XIRR中位数": median_row["XIRR"],
        "中位数对应开始日期": median_row["开始日期"],
        "中位数对应结束日期": median_row["结束日期"],
        "XIRR算术均值": np.mean(xirrs),
        "亏损窗口占比 (%)": loss_ratio,
        "10%分位数": np.percentile(xirrs, 10),
        "90%分位数": np.percentile(xirrs, 90),
        "XIRR标准差": np.std(xirrs, ddof=1),
        "最小XIRR": np.min(xirrs),
        "最大XIRR": np.max(xirrs),
    }

    # ---- 复合评分：遍历每个窗口计算得分，写回 results ----
    best_row = None
    best_score = -float("inf")

    for r in results:
        if np.isnan(r.get("XIRR", np.nan)):
            r["Z分数"] = np.nan
            r["稳定性系数"] = np.nan
            r["下行惩罚"] = np.nan
            r["综合评分"] = np.nan
            continue

        xirr = r["XIRR"]
        z = (xirr - median) / std

        if strategy == "conservative":
            stability = 1.0 / (1.0 + abs(z))
            loss_penalty = safe_ratio ** 3
            score = xirr * stability * loss_penalty

        elif strategy == "aggressive":
            stability = 1.0  # 不惩罚偏离
            loss_penalty = max(1.0 - loss_ratio / 200, 0.5)
            score = xirr * loss_penalty

        else:  # balanced
            if z >= 0:
                stability = 1.0 / (1.0 + max(z - 1.5, 0) * 0.3)
            else:
                stability = 1.0 / (1.0 + abs(z) * 0.6)
            loss_penalty = safe_ratio ** 2
            score = xirr * stability * loss_penalty

        # 写回窗口明细
        r["Z分数"] = round(z, 4)
        r["稳定性系数"] = round(stability, 4)
        r["下行惩罚"] = round(loss_penalty, 4)
        r["综合评分"] = round(score, 4)

        if score > best_score:
            best_score = score
            best_row = r

    # ---- 策略选取结果 ----
    stats["选取XIRR"] = best_row["XIRR"]
    stats["选取开始日期"] = best_row["开始日期"]
    stats["选取结束日期"] = best_row["结束日期"]
    stats["选取评分"] = round(best_score, 4)

    return stats



# ============================================================
# 5. 输出
# ============================================================
def print_stats(stats: dict, wy: int, tc: str, sn: str, mode: str = "monthly"):
    """打印统计摘要"""
    invest_desc = f"每月 {TARGET_DAY} 日" if mode == "monthly" else "每日定投"
    print(f"\n{'=' * 65}")
    print("📊 滚动窗口 XIRR 分布统计")
    print(f"{'=' * 65}")
    print(f"  ETF: {sn} ({tc})")
    print(f"  窗口: {wy} 年，步长 {STEP_MONTHS} 个月")
    print(f"  定投方式: {invest_desc}")
    print()

    rows = [
        ("📋 样本总数", stats.get("样本总数", 0)),
        ("📐 XIRR 中位数", stats.get("XIRR中位数", 0)),
        ("📅 中位数对应开始日期", stats.get("中位数对应开始日期", "-")),
        ("📅 中位数对应结束日期", stats.get("中位数对应结束日期", "-")),
        ("📐 XIRR 算术均值", stats.get("XIRR算术均值", 0)),
        ("📉 亏损窗口占比", stats.get("亏损窗口占比 (%)", 0)),
        ("📊 10% 分位数", stats.get("10%分位数", 0)),
        ("📊 90% 分位数", stats.get("90%分位数", 0)),
        ("📏 XIRR 标准差", stats.get("XIRR标准差", 0)),
        ("📉 最小 XIRR", stats.get("最小XIRR", 0)),
        ("📈 最大 XIRR", stats.get("最大XIRR", 0)),
    ]
    for label, value in rows:
        if isinstance(value, pd.Timestamp):
            print(f"  {label:<18}  {value.strftime('%Y-%m-%d')}")
        elif isinstance(value, float):
            print(f"  {label:<18}  {value:.2f}")
        else:
            print(f"  {label:<18}  {value}")


def plot_rolling(results: list, stats: dict, wy: int, tc: str, sn: str, strategy: str = "balanced", mode: str = "monthly"):
    """绘制 XIRR 滚动分布图"""
    df = pd.DataFrame(results).dropna(subset=["XIRR"])
    daily_suffix = "_daily" if mode == "daily" else ""
    if df.empty:
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10))

    # 上图：XIRR 时间序列
    ax1.plot(df["开始日期"], df["XIRR"], color="#3498db", linewidth=1.0)
    ax1.axhline(y=0, color="#e74c3c", linestyle="--", linewidth=0.8, alpha=0.6)
    ax1.axhline(y=stats["XIRR中位数"], color="#e67e22", linestyle="--", linewidth=1.0,
                label=f"中位数: {stats['XIRR中位数']:.2f}%")
    ax1.axhline(y=stats["XIRR算术均值"], color="#2ecc71", linestyle="--", linewidth=1.0,
                label=f"均值: {stats['XIRR算术均值']:.2f}%")
    ax1.axhline(y=stats["10%分位数"], color="#9b59b6", linestyle=":", linewidth=0.8, alpha=0.7)
    ax1.axhline(y=stats["90%分位数"], color="#9b59b6", linestyle=":", linewidth=0.8, alpha=0.7)
    ax1.fill_between(df["开始日期"], 0, df["XIRR"],
                     where=(df["XIRR"] >= 0), color="#3498db", alpha=0.15)
    ax1.fill_between(df["开始日期"], 0, df["XIRR"],
                     where=(df["XIRR"] < 0), color="#e74c3c", alpha=0.15)
    ax1.set_title(f"{sn} ({tc}) 滚动{wy}年定投 XIRR",
                  fontsize=14, fontweight="bold")
    ax1.set_ylabel("XIRR (%)", fontsize=11)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3, linestyle="--")

    # 下图：XIRR 分布直方图
    ax2.hist(df["XIRR"], bins=40, color="#3498db", edgecolor="white", alpha=0.8)
    ax2.axvline(x=0, color="#e74c3c", linestyle="--", linewidth=1.0, label="盈亏分界")
    ax2.axvline(x=stats["XIRR中位数"], color="#e67e22", linestyle="--", linewidth=1.5,
                label=f"中位数: {stats['XIRR中位数']:.2f}%")
    ax2.axvline(x=stats["XIRR算术均值"], color="#2ecc71", linestyle="--", linewidth=1.5,
                label=f"均值: {stats['XIRR算术均值']:.2f}%")
    ax2.axvline(x=stats["10%分位数"], color="#9b59b6", linestyle=":", linewidth=1.0,
                label=f"P10: {stats['10%分位数']:.2f}%")
    ax2.axvline(x=stats["90%分位数"], color="#9b59b6", linestyle=":", linewidth=1.0,
                label=f"P90: {stats['90%分位数']:.2f}%")
    ax2.set_xlabel("XIRR (%)", fontsize=11)
    ax2.set_ylabel("频数", fontsize=11)
    ax2.set_title(f"XIRR 分布（n={stats['样本总数']}）", fontsize=13, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3, linestyle="--", axis="y")

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"rolling_{wy}y_{sn}{daily_suffix}_{strategy}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n📈 图表已保存至：{out_path}")

# 查找指定滚动窗口期的最有预测价值的窗口
# 策略选取逻辑在 compute_stats 中实现
def pick_widows(df_full: DataFrame, wy: int = WINDOW_YEARS, tc: str = TS_CODE,
                sn: str = STOCK_NAME, strategy: str = "balanced"):
    # 3. 生成滚动窗口
    data_start = df_full["Date"].iloc[0].strftime("%Y%m%d")
    data_end = df_full["Date"].iloc[-1].strftime("%Y%m%d")
    print(f"\n📅 数据范围: {data_start} → {data_end}")
    windows = generate_rolling_windows(data_start, data_end, wy, STEP_MONTHS)
    if not windows:
        print(f"⚠️ 数据不足以生成 {wy} 年滚动窗口，跳过")
        return None, np.nan, None, None

    # 4. 滚动回测
    print(f"\n🔍 开始滚动回测（{wy}年窗口，步长{STEP_MONTHS}个月）...")
    results = run_rolling_backtest(df_full, windows)

    # 5. 统计 + 策略选取（compute_stats 内含分位数选取逻辑）
    stats = compute_stats(results, strategy)
    print_stats(stats, wy, tc, sn)

    if stats.get("样本总数", 0) == 0:
        return None, np.nan, None, None

    # 6. 保存 Excel
    xlsx_path = os.path.join(out_dir, f"rolling_{wy}y_{sn}_{strategy}.xlsx")
    result_df = pd.DataFrame(results)
    stats_rows = [{"指标": k, "数值": v} for k, v in stats.items()]
    stats_df = pd.DataFrame(stats_rows)
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        result_df.to_excel(writer, sheet_name="滚动窗口明细", index=False)
        stats_df.to_excel(writer, sheet_name="统计摘要", index=False)
    print(f"\n📄 Excel 已保存至：{xlsx_path}")

    # 7. 画图
    plot_rolling(results, stats, wy, tc, sn, strategy)
    plt.show()

    # 8. 从 stats 取策略选取结果，截取窗口数据
    start = stats["选取开始日期"]
    end = stats["选取结束日期"]
    pick_xirr = stats["选取XIRR"]

    mask = (df_full["Date"] >= start.strftime("%Y%m%d")) & \
           (df_full["Date"] <= end.strftime("%Y%m%d"))
    df_windows_pick = df_full[mask].copy()

    strategy_name = {"conservative": "保守型", "balanced": "均衡型", "aggressive": "激进型"}
    print(f"\n🎯 [{strategy_name.get(strategy, strategy)}] 选取窗口: "
          f"{start.strftime('%Y-%m-%d')} → {end.strftime('%Y-%m-%d')}  XIRR: {pick_xirr:.2f}%")
    return df_windows_pick, pick_xirr, start, end

# 每日定投版 pick_widows
def pick_widows_daily(df_full: DataFrame, wy: int = WINDOW_YEARS, tc: str = TS_CODE,
                      sn: str = STOCK_NAME, strategy: str = "balanced"):
    data_start = df_full["Date"].iloc[0].strftime("%Y%m%d")
    data_end = df_full["Date"].iloc[-1].strftime("%Y%m%d")
    print(f"\n📅 数据范围: {data_start} → {data_end}")
    windows = generate_rolling_windows(data_start, data_end, wy, STEP_MONTHS)
    if not windows:
        print(f"⚠️ 数据不足以生成 {wy} 年滚动窗口，跳过")
        return None, np.nan, None, None

    print(f"\n🔍 开始滚动回测（{wy}年窗口，步长{STEP_MONTHS}个月，每日定投）...")
    results = run_rolling_backtest_daily(df_full, windows)

    stats = compute_stats(results, strategy)
    print_stats(stats, wy, tc, sn, mode="daily")

    if stats.get("样本总数", 0) == 0:
        return None, np.nan, None, None

    # 保存 Excel
    xlsx_path = os.path.join(out_dir, f"rolling_{wy}y_{sn}_daily_{strategy}.xlsx")
    result_df = pd.DataFrame(results)
    stats_rows = [{"指标": k, "数值": v} for k, v in stats.items()]
    stats_df = pd.DataFrame(stats_rows)
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        result_df.to_excel(writer, sheet_name="滚动窗口明细", index=False)
        stats_df.to_excel(writer, sheet_name="统计摘要", index=False)
    print(f"\n📄 Excel 已保存至：{xlsx_path}")

    plot_rolling(results, stats, wy, tc, sn, strategy, mode="daily")
    plt.show()

    # 从 stats 取策略选取结果
    start = stats["选取开始日期"]
    end = stats["选取结束日期"]
    pick_xirr = stats["选取XIRR"]

    mask = (df_full["Date"] >= start.strftime("%Y%m%d")) & \
           (df_full["Date"] <= end.strftime("%Y%m%d"))
    df_windows_pick = df_full[mask].copy()

    strategy_name = {"conservative": "保守型", "balanced": "均衡型", "aggressive": "激进型"}
    print(f"\n🎯 [{strategy_name.get(strategy, strategy)}] 选取窗口: "
          f"{start.strftime('%Y-%m-%d')} → {end.strftime('%Y-%m-%d')}  XIRR: {pick_xirr:.2f}%")
    return df_windows_pick, pick_xirr, start, end

def fetch_full_data(tc:str =TS_CODE,sn:str =STOCK_NAME) -> DataFrame:
    # 1. 获取 ETF 上市日期
    list_date = get_etf_list_date(tc, sn)

    # 2. 获取全部历史数据
    today = pd.Timestamp.today().strftime("%Y%m%d")
    print(f"⏳ 正在获取 {sn} ({tc}) 全量历史数据...")
    config = BacktestConfig(
        ts_code=tc, stock_name=sn,
        market="ETF", start_date=list_date,
        end_date=today,
    )
    df_full = fetch_data(
        ts_code=config.ts_code, stock_name=config.stock_name,
        market=config.market, start_date=config.start_date,
        end_date=config.end_date, exchange=config.exchange,
    )
    return df_full

# ============================================================
# Main
# ============================================================
def main():
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "PingFang SC"]
    plt.rcParams["axes.unicode_minus"] = False

    df_full = fetch_full_data()
    pick_widows(df_full)


if __name__ == "__main__":
    main()
