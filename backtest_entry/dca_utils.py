"""DCA 定投回测工具函数 —— 供 run_dca_backtest / optimize_dca_day / rolling_dca_backtest 共用"""

import pandas as pd
import numpy as np


# ============================================================
# 1. 定位每月实际投资日
# ============================================================
def find_investment_dates(df: pd.DataFrame, target_day: int) -> pd.DataFrame:
    """找到每月 >=target_day 的第一个交易日，返回 (Date, Close, year_month)"""
    df = df.copy()
    df["year_month"] = df["Date"].dt.to_period("M")
    df["day"] = df["Date"].dt.day

    # 每月取 >= target_day 的第一个交易日
    eligible = df[df["day"] >= target_day]
    invest_dates = eligible.groupby("year_month").first().reset_index()
    # 合并回完整的 Date 列
    result = df[df["Date"].isin(invest_dates["Date"])][["Date", "Close", "year_month"]].copy()
    return result.sort_values("Date").reset_index(drop=True)


# ============================================================
# 2. DCA 模拟引擎
# ============================================================
def simulate_dca(df: pd.DataFrame, invest_dates: pd.DataFrame,
                 monthly_amount: float, per_shares: int) -> dict:
    """执行定投模拟，返回权益曲线和每期投资明细"""
    shares = 0          # 累计份额（股）
    cash = 0.0          # 剩余现金
    total_invested = 0.0
    investments = []    # 每期投资记录

    for _, row in invest_dates.iterrows():
        date = row["Date"]
        close = row["Close"]
        available = monthly_amount + cash
        lot_price = close * per_shares

        max_lots = int(available / lot_price)
        if max_lots > 0:
            cost = max_lots * lot_price
            shares += max_lots * per_shares
            cash = available - cost
            total_invested += cost
        else:
            cost = 0
            cash = available

        investments.append({
            "投资日期": date.strftime("%Y-%m-%d"),
            "价格": round(close, 4),
            "买入手数": max_lots,
            "买入金额": round(cost, 2),
            "累计份额": shares,
            "剩余现金": round(cash, 2),
            "累计投入": round(total_invested, 2),
        })

    # —— 构建每日权益曲线 ——
    df_inv = pd.DataFrame(investments)
    df_inv["投资日期"] = pd.to_datetime(df_inv["投资日期"])
    # 构建投资日期 → (累计份额, 剩余现金) 的查找字典
    inv_lookup = dict(zip(df_inv["投资日期"],
                          zip(df_inv["累计份额"], df_inv["剩余现金"])))

    df_sorted = df.sort_values("Date")
    equity_curve = []
    equity_dates = []
    for _, r in df_sorted.iterrows():
        d = r["Date"]
        # 找到该日期之前（含）最近一次投资的累计份额和剩余现金
        past_dates = [t for t in inv_lookup if t <= d]
        if past_dates:
            latest = max(past_dates)
            cum_shares, rem_cash = inv_lookup[latest]
        else:
            cum_shares, rem_cash = 0, 0.0
        value = cum_shares * r["Close"] + rem_cash
        equity_curve.append(value)
        equity_dates.append(d)

    equity_series = pd.Series(equity_curve, index=equity_dates)

    return {
        "investments": pd.DataFrame(investments),
        "equity_series": equity_series,
        "equity_dates": pd.Series(equity_dates),
        "total_invested": total_invested,
        "total_shares": shares,
        "final_cash": cash,
        "final_value": shares * df_sorted["Close"].iloc[-1] + cash,
        "last_close": df_sorted["Close"].iloc[-1],
    }


# ============================================================
# 3. XIRR 计算
# ============================================================
def _compute_xirr(cashflows: list, dates: list, guess: float = 0.1) -> float:
    """计算 XIRR（内部收益率），使用牛顿迭代法

    Args:
        cashflows: 现金流列表（负数为投入，正数为收回）
        dates: 对应的日期列表
        guess: 初始猜测值

    Returns:
        年化收益率（小数形式），计算失败返回 0
    """
    dates = pd.to_datetime(dates)
    d0 = dates[0]
    years = np.array([(d - d0).days / 365.0 for d in dates])
    cf = np.array(cashflows, dtype=float)

    # 尝试多个初始值，避免牛顿法不收敛
    for rate in [guess, -0.5, -0.3, 0.0, 0.2, 0.5, 1.0, 2.0]:
        r = max(rate, -0.99)  # 守卫：r 必须 > -1，否则 (1+r) 为负导致 power 报错
        for _ in range(200):
            discount = (1 + r) ** years
            npv_val = np.sum(cf / discount)
            dnpv = np.sum(-years * cf / ((1 + r) ** (years + 1)))
            if abs(dnpv) < 1e-12:
                break
            new_r = r - npv_val / dnpv
            # 牛顿步长可能越界，限制在有效范围内
            new_r = max(new_r, -0.99)
            if abs(new_r - r) < 1e-8:
                return new_r
            r = new_r

    return 0.0
