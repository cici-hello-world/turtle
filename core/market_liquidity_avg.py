#!/usr/bin/env python3
"""全市场股票日均成交额统计 —— A股 + 港股

功能：
  拉取所有正常交易股票的近1年/5年日均成交额，按市场聚合，保存 Excel。

用法：
  python core/market_liquidity_avy.py          # 默认: 1年 + 5年
  python core/market_liquidity_avy.py 1y       # 仅近1年
  python core/market_liquidity_avy.py 5y       # 仅近5年

输出：
  market_liquidity_avy_1y.xlsx  — 近1年日均成交额排名
  market_liquidity_avy_5y.xlsx  — 近5年日均成交额排名

实现策略：
  - A股: pro.monthly(trade_date=月末) → 月成交额 ÷ 当月交易天数 = 日均
  - 港股: pro.hk_daily 逐日拉取 → 累加成交额 ÷ 有成交天数 = 日均
  - 请求间自动限速，异常自动重试

依赖：
  tushare, pandas, openpyxl
  环境变量 TUSHARE_TOKEN
"""

import os
import sys
import time
import pandas as pd
import tushare as ts
from datetime import datetime, timedelta

# ============================================================
# 配置
# ============================================================
PERIODS = []
if len(sys.argv) > 1:
    for a in sys.argv[1:]:
        if a == "1y":
            PERIODS.append(("1y", 1))
        elif a == "5y":
            PERIODS.append(("5y", 5))
if not PERIODS:
    PERIODS = [("1y", 1), ("5y", 5)]

RATE_LIMIT = 0.35   # 请求间隔（秒）
MAX_RETRIES = 3     # 请求最大重试次数

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(OUTPUT_DIR)


# ============================================================
# 工具
# ============================================================
def _date_range_years(years: int) -> tuple[str, str]:
    """返回 YYYYMMDD 格式的起止日期"""
    end = datetime.today()
    start = end - timedelta(days=years * 365)
    return start.strftime("%Y%m%d"), end.strftime("%Y%m%d")


def get_a_codes(pro) -> list[str]:
    """获取 A 股正常交易股票代码（排除 ST）"""
    df = pro.stock_basic(list_status="L", fields="ts_code,name")
    mask_st = df["name"].str.contains(r"\*?ST", na=False)
    codes = df[~mask_st]["ts_code"].tolist()
    print(f"  A股正常交易: {len(codes)} 只 (排除 ST {mask_st.sum()} 只)")
    return codes


def get_hk_codes(pro) -> list[str]:
    """获取港股正常交易股票代码"""
    df = pro.hk_basic(list_status="L", fields="ts_code,name")
    print(f"  港股正常交易: {len(df)} 只")
    return df["ts_code"].tolist()


def generate_month_end_dates(pro, exchange: str,
                             start: str, end: str) -> list[str]:
    """从交易日历中提取每月最后一个交易日（YYYYMMDD）"""
    cal = pro.trade_cal(exchange=exchange, start_date=start, end_date=end)
    open_days = cal[cal["is_open"] == 1].copy()
    if open_days.empty:
        return []
    open_days["dt"] = pd.to_datetime(open_days["cal_date"])
    open_days["year_month"] = open_days["dt"].dt.to_period("M")
    # 用 max 取每月最大日期（日历数据可能降序排列）
    month_ends = open_days.groupby("year_month")["cal_date"].max()
    return sorted(month_ends.tolist())


def fetch_trading_days(pro, exchange: str, start: str, end: str) -> list[str]:
    """获取交易日列表（YYYYMMDD）"""
    cal = pro.trade_cal(exchange=exchange, start_date=start, end_date=end)
    days = cal[cal["is_open"] == 1]["cal_date"].tolist()
    return sorted(days)


def get_trading_days_in_month(pro, exchange: str,
                              start: str, end: str) -> dict[str, int]:
    """获取每月交易天数 {YYYYMM: trading_days_count}"""
    cal = pro.trade_cal(exchange=exchange, start_date=start, end_date=end)
    open_days = cal[cal["is_open"] == 1].copy()
    open_days["dt"] = pd.to_datetime(open_days["cal_date"])
    open_days["ym"] = open_days["dt"].dt.strftime("%Y%m")
    counts = open_days.groupby("ym").size()
    return counts.to_dict()


# ============================================================
# 数据拉取
# ============================================================
def fetch_a_monthly(pro, trade_date: str) -> pd.DataFrame | None:
    """拉取 A 股单月全量月线（monthly 接口）"""
    for attempt in range(MAX_RETRIES):
        try:
            df = pro.monthly(trade_date=trade_date)
            if df is not None and len(df) > 0:
                return df[["ts_code", "amount"]]
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"    ⚠️ A股 monthly({trade_date}) 失败: {e}")
    return None


def fetch_hk_daily(pro, trade_date: str) -> pd.DataFrame | None:
    """拉取港股单个交易日全量日线"""
    for attempt in range(MAX_RETRIES):
        try:
            df = pro.hk_daily(start_date=trade_date, end_date=trade_date)
            if df is not None and len(df) > 0:
                return df[["ts_code", "amount"]]
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"    ⚠️ 港股 hk_daily({trade_date}) 失败: {e}")
    return None


# ============================================================
# A股：逐月 monthly 累加
# ============================================================
def compute_avg_amount_a(pro, codes: list[str],
                         month_dates: list[str],
                         td_counts: dict[str, int],
                         label: str) -> pd.DataFrame:
    """
    A股：逐月拉取 monthly(trade_date)，累加月成交额，
    日均 = 累计月成交额 ÷ 累计有成交天数。
    """
    accum_amount: dict[str, float] = {}
    accum_days: dict[str, int] = {}
    total_months = len(month_dates)
    success = 0

    print(f"\n📡 逐月拉取{label} A股 monthly 数据 "
          f"({total_months} 个月)")

    for i, md in enumerate(month_dates, 1):
        df = fetch_a_monthly(pro, md)
        time.sleep(RATE_LIMIT)

        if df is None or df.empty:
            continue
        success += 1

        ym = md[:6]
        tdays = td_counts.get(ym, 21)

        for _, r in df.iterrows():
            code = r["ts_code"]
            amt = r["amount"]
            if pd.isna(amt) or amt <= 0:
                continue
            accum_amount[code] = accum_amount.get(code, 0.0) + float(amt)
            accum_days[code] = accum_days.get(code, 0) + tdays

        if i % 6 == 0 or i == total_months:
            pct = i / total_months * 100
            print(f"  [{i}/{total_months}] {pct:.0f}%  成功 {success} 月  "
                  f"已覆盖 {len(accum_amount)} 只股票")

    # 构建结果
    rows = []
    for code in codes:
        total_amt = accum_amount.get(code, 0.0)
        days = accum_days.get(code, 0)
        avg_amt = total_amt / days if days > 0 else 0.0
        rows.append({
            "ts_code": code,
            "months_with_data": success,
            "trading_days": days,
            "avg_daily_amount": avg_amt,
        })

    result = pd.DataFrame(rows)
    result = result.sort_values("avg_daily_amount", ascending=False).reset_index(drop=True)
    result["rank"] = range(1, len(result) + 1)

    basic = pro.stock_basic(list_status="L", fields="ts_code,name")
    name_map = dict(zip(basic["ts_code"], basic["name"]))
    result["name"] = result["ts_code"].map(name_map)
    return result


# ============================================================
# 港股：逐日 daily 累加
# ============================================================
def compute_avg_amount_hk(pro, codes: list[str],
                          trading_days: list[str],
                          label: str) -> pd.DataFrame:
    """
    港股：逐日拉取 hk_daily，累加每只股票的成交额，
    日均 = 累计成交额 ÷ 有成交天数。
    """
    accum_amount: dict[str, float] = {}
    accum_days: dict[str, int] = {}
    total_days = len(trading_days)
    success = 0

    print(f"\n📡 逐日拉取{label}港股 daily 数据 "
          f"({total_days} 个交易日)")

    for i, td in enumerate(trading_days, 1):
        df = fetch_hk_daily(pro, td)
        time.sleep(RATE_LIMIT)

        if df is None or df.empty:
            continue
        success += 1

        for _, r in df.iterrows():
            code = r["ts_code"]
            amt = r["amount"]
            if pd.isna(amt) or amt <= 0:
                continue
            accum_amount[code] = accum_amount.get(code, 0.0) + float(amt)
            accum_days[code] = accum_days.get(code, 0) + 1

        if i % 50 == 0 or i == total_days:
            pct = i / total_days * 100
            print(f"  [{i}/{total_days}] {pct:.0f}%  成功 {success} 天  "
                  f"已覆盖 {len(accum_amount)} 只股票")

    # 构建结果（港股用交易天数，不同于A股月数）
    rows = []
    for code in codes:
        total_amt = accum_amount.get(code, 0.0)
        days = accum_days.get(code, 0)
        avg_amt = total_amt / days if days > 0 else 0.0
        rows.append({
            "ts_code": code,
            "months_with_data": days,  # 复用字段名，港股含义为有成交天数
            "avg_daily_amount": avg_amt,
        })

    result = pd.DataFrame(rows)
    result = result.sort_values("avg_daily_amount", ascending=False).reset_index(drop=True)
    result["rank"] = range(1, len(result) + 1)

    basic = pro.hk_basic(list_status="L", fields="ts_code,name")
    name_map = dict(zip(basic["ts_code"], basic["name"]))
    result["name"] = result["ts_code"].map(name_map)
    return result


# ============================================================
# 保存
# ============================================================
def save_to_excel(df_a: pd.DataFrame, df_hk: pd.DataFrame,
                  years: int, output_path: str):
    """保存 A 股 + 港股各一个 sheet"""
    label = f"近{years}年"

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        # A股
        a_out = df_a[["rank", "ts_code", "name", "avg_daily_amount",
                       "trading_days"]].copy()
        a_out.columns = ["排名", "代码", "名称", "日均成交额", "累计交易天数"]
        a_out.to_excel(writer, sheet_name=f"A股{label}", index=False)

        # 港股
        hk_out = df_hk[["rank", "ts_code", "name", "avg_daily_amount",
                        "months_with_data"]].copy()
        hk_out.columns = ["排名", "代码", "名称", "日均成交额", "有成交天数"]
        hk_out.to_excel(writer, sheet_name=f"港股{label}", index=False)

        # 汇总
        summary_data = [
            ["市场", "股票数", "日均成交额合计", "平均每只日均成交额"],
            ["A股", len(df_a),
             f"{df_a['avg_daily_amount'].sum():,.0f}",
             f"{df_a['avg_daily_amount'].mean():,.0f}"],
            ["港股", len(df_hk),
             f"{df_hk['avg_daily_amount'].sum():,.0f}",
             f"{df_hk['avg_daily_amount'].mean():,.0f}"],
        ]
        pd.DataFrame(summary_data[1:], columns=summary_data[0]).to_excel(
            writer, sheet_name="汇总", index=False
        )

    file_size = os.path.getsize(output_path) / 1024
    print(f"\n✅ 已保存 {output_path} ({file_size:.0f} KB)")

    # TOP10 预览
    print(f"\n{'='*60}")
    print(f"📊 {label}日均成交额 TOP10")
    print(f"{'='*60}")
    for mkt, d, col, unit in [("A股", df_a, "trading_days", "天"),
                                ("港股", df_hk, "months_with_data", "天")]:
        print(f"\n  【{mkt}】")
        for _, r in d.head(10).iterrows():
            print(f"  {r['rank']:>3}. {r['ts_code']} {r['name']:<8}  "
                  f"日均: {r['avg_daily_amount']:>15,.0f}  "
                  f"({r[col]}{unit})")


# ============================================================
# 主入口
# ============================================================
def main():
    token = os.environ.get("TUSHARE_TOKEN")
    if not token:
        print("❌ 请设置 TUSHARE_TOKEN")
        return

    ts.set_token(token)
    pro = ts.pro_api()

    # ———— 获取股票列表 ————
    print("=" * 60)
    print("📋 获取全市场股票列表")
    print("=" * 60)
    a_codes = get_a_codes(pro)
    hk_codes = get_hk_codes(pro)

    for period_label, years in PERIODS:
        start, end = _date_range_years(years)
        print(f"\n{'='*60}")
        print(f"📊 统计周期: 近{years}年 ({start} → {end})")
        print(f"{'='*60}")

        # ———— A股：月频 ————
        a_month_dates = generate_month_end_dates(pro, "SSE", start, end)
        td_a = get_trading_days_in_month(pro, "SSE", start, end)
        print(f"  A股月末数: {len(a_month_dates)} 个月")

        # ———— 港股：日频 ————
        hk_days = fetch_trading_days(pro, "XHKG", start, end)
        print(f"  港股交易日: {len(hk_days)} 天")

        # ———— 拉取并计算 ————
        df_a = compute_avg_amount_a(pro, a_codes, a_month_dates, td_a,
                                    f"A股近{years}年")
        df_hk = compute_avg_amount_hk(pro, hk_codes, hk_days,
                                      f"港股近{years}年")

        # ———— 保存 ————
        output_name = f"market_liquidity_avg_{period_label}.xlsx"
        output_path = os.path.join(PROJECT_DIR, output_name)
        save_to_excel(df_a, df_hk, years, output_path)


if __name__ == "__main__":
    main()
