#!/usr/bin/env python3
"""全市场股票当日成交额排名 —— A股 + 港股

功能：
  拉取所有正常交易股票的最新交易日成交额，按市场聚合排名，保存 Excel。

用法：
  python core/market_liquidity.py

输出：
  market_liquidity.xlsx — 当日成交额排名

实现策略：
  - A股: pro.daily(start_date=最新交易日, end_date=最新交易日)
  - 港股: pro.hk_daily(start_date=最新交易日, end_date=最新交易日)
  - 一次调用覆盖全量股票

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
MAX_RETRIES = 3
RATE_LIMIT = 0.35

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(OUTPUT_DIR)


# ============================================================
# 工具
# ============================================================
def get_latest_trade_date(pro, exchange: str) -> str:
    """获取最近一个交易日"""
    today = (datetime.today() - timedelta(days=2)).strftime("%Y%m%d")
    cal = pro.trade_cal(exchange=exchange,
                        start_date=(datetime.today() - timedelta(days=10)).strftime("%Y%m%d"),
                        end_date=today)
    open_days = cal[cal["is_open"] == 1]["cal_date"].tolist()
    return max(open_days) if open_days else today


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


# ============================================================
# 数据拉取
# ============================================================
def fetch_a_amount(pro, trade_date: str) -> pd.DataFrame | None:
    """拉取 A 股单日全量成交额"""
    for attempt in range(MAX_RETRIES):
        try:
            df = pro.daily(start_date=trade_date, end_date=trade_date)
            if df is not None and len(df) > 0:
                return df[["ts_code", "amount","pct_chg"]]
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"    ⚠️ A股 daily({trade_date}) 失败: {e}")
    return None


def fetch_hk_amount(pro, trade_date: str) -> pd.DataFrame | None:
    """拉取港股单日全量成交额"""
    for attempt in range(MAX_RETRIES):
        try:
            df = pro.hk_daily(start_date=trade_date, end_date=trade_date)
            if df is not None and len(df) > 0:
                return df[["ts_code", "amount", "pct_chg"]]
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"    ⚠️ 港股 hk_daily({trade_date}) 失败: {e}")
    return None


# ============================================================
# 主逻辑
# ============================================================
def build_ranking(df_amount: pd.DataFrame, pro, market: str) -> pd.DataFrame:
    """构建排名 DataFrame"""
    df = df_amount.copy()
    df = df[df["amount"].notna() & (df["amount"] > 0)]
    df = df.sort_values("amount", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    df.columns = ["ts_code", "amount","pct_chg", "rank"]

    # 添加名称
    if market == "A":
        basic = pro.stock_basic(list_status="L", fields="ts_code,name")
    else:
        basic = pro.hk_basic(list_status="L", fields="ts_code,name")
    name_map = dict(zip(basic["ts_code"], basic["name"]))
    df["name"] = df["ts_code"].map(name_map)

    return df[["rank", "ts_code", "name", "amount", "pct_chg"]]


# ============================================================
# 保存
# ============================================================
def save_to_excel(df_a: pd.DataFrame, df_hk: pd.DataFrame,
                  trade_date: str, output_path: str):
    """保存 A 股 + 港股各一个 sheet"""
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        a_out = df_a.copy()
        a_out.columns = ["排名", "代码", "名称", "成交额", "涨跌幅"]
        a_out.to_excel(writer, sheet_name="A股", index=False)

        hk_out = df_hk.copy()
        hk_out.columns = ["排名", "代码", "名称", "成交额", "涨跌幅"]
        hk_out.to_excel(writer, sheet_name="港股", index=False)

        # 汇总
        summary_data = [
            ["市场", "交易日", "股票数", "成交额合计", "平均每只成交额"],
            ["A股", trade_date, len(df_a),
             f"{df_a['amount'].sum():,.0f}",
             f"{df_a['amount'].mean():,.0f}"],
            ["港股", trade_date, len(df_hk),
             f"{df_hk['amount'].sum():,.0f}",
             f"{df_hk['amount'].mean():,.0f}"],
        ]
        pd.DataFrame(summary_data[1:], columns=summary_data[0]).to_excel(
            writer, sheet_name="汇总", index=False
        )

    print(f"\n✅ 已保存 {output_path}")

    # TOP10 预览
    print(f"\n{'='*60}")
    print(f"📊 交易日 {trade_date} 成交额 TOP10")
    print(f"{'='*60}")
    for mkt, d in [("A股", df_a), ("港股", df_hk)]:
        print(f"\n  【{mkt}】")
        for _, r in d.head(10).iterrows():
            print(f"  {r['rank']:>3}. {r['ts_code']} {r['name']:<8}  "
                  f"成交额: {r['amount']:>15,.0f}")


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

    # ———— 获取最新交易日 ————
    a_date = get_latest_trade_date(pro, "SSE")
    hk_date = get_latest_trade_date(pro, "XHKG")
    print(f"A股最新交易日: {a_date}  |  港股最新交易日: {hk_date}")

    # ———— 拉取当日成交额 ————
    print("\n📡 拉取 A 股当日成交额 ...")
    df_a = fetch_a_amount(pro, a_date)
    time.sleep(RATE_LIMIT)

    print("📡 拉取港股当日成交额 ...")
    df_hk = fetch_hk_amount(pro, hk_date)

    if df_a is None and df_hk is None:
        print("❌ 两市场数据拉取均失败")
        return

    # ———— 构建排名 ————
    rank_a = build_ranking(df_a, pro, "A") if df_a is not None else pd.DataFrame()
    rank_hk = build_ranking(df_hk, pro, "HK") if df_hk is not None else pd.DataFrame()

    # ———— 保存 ————
    OUTPUT_PATH = os.path.join(PROJECT_DIR, f"market_liquidity_{a_date}.xlsx")
    save_to_excel(rank_a, rank_hk, a_date, OUTPUT_PATH)


if __name__ == "__main__":
    main()
