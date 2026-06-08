#!/usr/bin/env python3
"""唐奇安 EMA 趋势过滤器参数优化 —— 基于流动性 TOP10 加仓做多回测

功能：
  1. 筛选近五年日均成交额最高的 10 只股票 (港股/ A股)
  2. 对每只股票运行唐奇安通道突破 + 金字塔加仓回测
  3. 对比不同 EMA 快/慢线参数组合的效果
  4. 输出综合排名与最优参数建议

用法：
  python optimize_donchian_ema.py          # 默认港股
  python optimize_donchian_ema.py A        # A股

依赖：
  tushare, akshare (仅港股分红补权), pandas, numpy
  环境变量 TUSHARE_TOKEN 需要已配置
"""

import os
import re
import sys
import pandas as pd
import numpy as np
from itertools import product
import warnings
import tushare as ts

warnings.filterwarnings("ignore")

# ============================================================
# 参数配置
# ============================================================
# 市场切换: 环境变量 TURTLE_MARKET=A 或命令行传入 "A"
_raw = sys.argv[1].upper() if len(sys.argv) > 1 else os.environ.get("TURTLE_MARKET", "HK")
MARKET = _raw if _raw in ("HK", "A") else "HK"

START_DATE = "20210601"
END_DATE = "20260608"
INITIAL_CAPITAL = 200000.0
RISK_PERCENT = 1.0
N_VALUE = 2.0
PER_SHARES = 100
MAX_PYRAMID = 5
SKIP_FIRST_ADD = False

ENTRY_PERIOD = 20
EXIT_PERIOD = 10
ATR_PERIOD = 14

FAST_PERIODS = [20, 25, 30, 40, 50]
SLOW_PERIODS = [50, 60, 80, 100, 150, 200, 250, 300, 350]

# 港股流动性候选标的（覆盖主要蓝筹）
HK_CANDIDATES = [
    "00700.HK",  # 腾讯控股
    "09988.HK",  # 阿里巴巴
    "00941.HK",  # 中国移动
    "02318.HK",  # 中国平安
    "00388.HK",  # 香港交易所
    "01299.HK",  # 友邦保险
    "00005.HK",  # 汇丰控股
    "03690.HK",  # 美团
    "01810.HK",  # 小米集团
    "09999.HK",  # 网易
    "09618.HK",  # 京东集团
    "09888.HK",  # 百度集团
    "02015.HK",  # 理想汽车
    "01024.HK",  # 快手
    "02269.HK",  # 药明生物
    "00939.HK",  # 建设银行
    "01398.HK",  # 工商银行
    "03988.HK",  # 中国银行
    "01288.HK",  # 农业银行
    "00883.HK",  # 中国海洋石油
    "00857.HK",  # 中国石油股份
    "00386.HK",  # 中国石油化工
    "01093.HK",  # 石药集团
    "02020.HK",  # 安踏体育
    "09633.HK",  # 农夫山泉
    "02331.HK",  # 李宁
    "00175.HK",  # 吉利汽车
    "01211.HK",  # 比亚迪股份
    "09868.HK",  # 小鹏汽车
    "09961.HK",  # 携程集团
]

# A股蓝筹候选标的 (沪深300核心 + 行业龙头)
A_CANDIDATES = [
    # 金融
    "601318.SH",  # 中国平安
    "600036.SH",  # 招商银行
    "601166.SH",  # 兴业银行
    "600016.SH",  # 民生银行
    "601398.SH",  # 工商银行
    "601288.SH",  # 农业银行
    "601939.SH",  # 建设银行
    "600030.SH",  # 中信证券
    "601688.SH",  # 华泰证券
    "601628.SH",  # 中国人寿
    # 消费
    "600519.SH",  # 贵州茅台
    "000858.SZ",  # 五粮液
    "002304.SZ",  # 洋河股份
    "000568.SZ",  # 泸州老窖
    "600887.SH",  # 伊利股份
    "002714.SZ",  # 牧原股份
    "000895.SZ",  # 双汇发展
    # 科技
    "002415.SZ",  # 海康威视
    "000725.SZ",  # 京东方A
    "002475.SZ",  # 立讯精密
    "300750.SZ",  # 宁德时代
    "002371.SZ",  # 北方华创
    "688981.SH",  # 中芯国际
    # 新能源
    "002594.SZ",  # 比亚迪
    "601012.SH",  # 隆基绿能
    "300274.SZ",  # 阳光电源
    "600438.SH",  # 通威股份
    # 医药
    "600276.SH",  # 恒瑞医药
    "000538.SZ",  # 云南白药
    "300760.SZ",  # 迈瑞医疗
    "300122.SZ",  # 智飞生物
    # 工业/制造
    "600031.SH",  # 三一重工
    "000651.SZ",  # 格力电器
    "000333.SZ",  # 美的集团
    "600690.SH",  # 海尔智家
    "601899.SH",  # 紫金矿业
    "600585.SH",  # 海螺水泥
    # 能源
    "601857.SH",  # 中国石油
    "600028.SH",  # 中国石化
    "601088.SH",  # 中国神华
    # 通信
    "600050.SH",  # 中国联通
    "600941.SH",  # 中国移动
    "601728.SH",  # 中国电信
    # 汽车
    "600104.SH",  # 上汽集团
    "601633.SH",  # 长城汽车
    "000625.SZ",  # 长安汽车
    # 地产/基建
    "000002.SZ",  # 万科A
    "600048.SH",  # 保利发展
    "601668.SH",  # 中国建筑
    "601390.SH",  # 中国中铁
]

# 输出路径
_suffix = "a" if MARKET == "A" else "hk"
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), f"donchian_param_analysis_pyramid_{_suffix}.csv")


# ============================================================
# A股: adj_factor 前复权
# ============================================================
if MARKET == "A":
    import akshare as _dummy  # noqa (仅港股需要)

    def fetch_adj_factor(pro, ts_code: str) -> pd.DataFrame:
        """拉取 A 股复权因子"""
        return pro.adj_factor(ts_code=ts_code)

    def apply_adj_factor_a(df: pd.DataFrame,
                            adj: pd.DataFrame) -> pd.DataFrame:
        """用 adj_factor 生成 A 股前复权 OHLC"""
        df = df.merge(adj[['trade_date', 'adj_factor']], on='trade_date', how='left')
        # adj_factor 从后往前填充
        last = 1.0
        filled = []
        for i in range(len(df) - 1, -1, -1):
            v = df['adj_factor'].iloc[i]
            if pd.notna(v):
                last = float(v)
            filled.append(last)
        filled.reverse()
        df['adj_factor'] = filled
        for col in ['close', 'high', 'low', 'open']:
            if col in df.columns:
                df[col] = (df[col] * df['adj_factor']).round(4)
        return df

    def fetch_top10_by_liquidity(pro, candidates: list[str],
                                  start: str, end: str):
        """A股: daily + adj_factor → 前复权 → 按日均成交额排序 TOP10"""
        rank = []
        for ts_code in candidates:
            try:
                # 拉不复权日线
                df = pro.daily(ts_code=ts_code, start_date=start, end_date=end)
                if df is None or df.empty:
                    continue
                df["trade_date"] = pd.to_datetime(df["trade_date"])
                df = df.sort_values("trade_date").reset_index(drop=True)

                # 拉 adj_factor 并计算前复权
                adj = fetch_adj_factor(pro, ts_code)
                adj["trade_date"] = pd.to_datetime(adj["trade_date"])
                df = apply_adj_factor_a(df, adj)

                avg_amt = df["amount"].mean()
                rank.append((ts_code, avg_amt, df))
            except Exception as e:
                pass
        rank.sort(key=lambda x: x[1], reverse=True)
        return rank[:10]

else:
    # 港股: hk_daily_adj + akshare 补权
    import akshare as ak

    # ============================================================
    # 港股: 分红解析 & adj_factor 补权
    # ============================================================
    def parse_dividend_hkd(scheme: str) -> float | None:
        """从 akshare 分红方案文本中提取港币每股现金分红"""
        if not isinstance(scheme, str):
            return None
        m = re.search(r'相当于港币([\d.]+)元', scheme)
        if m: return float(m.group(1))
        m = re.search(r'每股派港币([\d.]+)元', scheme)
        if m: return float(m.group(1))
        return None


    def find_missing_dividends(df_adj: pd.DataFrame, code: str) -> list[tuple[str, float]]:
        """对比 Tushare adj_factor 最后变动日期与 akshare 分红记录"""
        d = df_adj.copy()
        d['adj_diff'] = d['adj_factor'].diff().abs()
        changes = d[d['adj_diff'] > 1e-8]
        last_adj_date = str(changes['trade_date'].iloc[-1])[:10].replace('-', '') \
            if not changes.empty else None

        ak_code = code.replace('.HK', '')
        try:
            div = ak.stock_hk_dividend_payout_em(symbol=ak_code)
        except Exception:
            return []

        missing = []
        for _, r in div.iterrows():
            ex_date_str = str(r['除净日']).replace('-', '')[:8]
            scheme = str(r['分红方案'])
            if ex_date_str > END_DATE:
                continue
            if '分派' in scheme and '每股派' not in scheme:
                continue
            if '送股' in scheme or '转股' in scheme:
                continue
            div_hkd = parse_dividend_hkd(scheme)
            if div_hkd is None or div_hkd <= 0:
                continue
            is_missing = (last_adj_date is None) or (ex_date_str > last_adj_date)
            if is_missing:
                missing.append((ex_date_str, div_hkd))
        return missing


    def apply_missing_adj(df: pd.DataFrame,
                          missing: list[tuple[str, float]]) -> pd.DataFrame:
        """对缺失的除权日，计算并应用 adj_factor 修正"""
        df = df.copy()
        df['adj_factor_corrected'] = df['adj_factor'].astype(float)
        n = len(df)
        for ex_date_str, div_hkd in missing:
            ex_dt = pd.Timestamp(ex_date_str)
            pos = None
            for i in range(n):
                if df.iloc[i]['trade_date'] >= ex_dt:
                    pos = i
                    break
            if pos is None:
                continue
            close_ex = float(df.iloc[pos]['close'])
            if pd.isna(close_ex) or close_ex <= 0:
                continue
            correction = close_ex / (close_ex + div_hkd)
            if pos > 0:
                col_idx = df.columns.get_loc('adj_factor_corrected')
                df.iloc[:pos, col_idx] *= correction
        return df


    def hk_compute_adjusted_prices(df: pd.DataFrame) -> pd.DataFrame:
        """用 adj_factor_corrected 生成前复权 OHLC"""
        adj = df['adj_factor_corrected'].values
        for col in ['close', 'high', 'low', 'open']:
            if col in df.columns:
                df[col] = (df[col] * adj).round(4)
        return df


    def fetch_top10_by_liquidity(pro, candidates: list[str],
                                  start: str, end: str):
        """港股: hk_daily_adj + akshare补权 → 按日均成交额排序 TOP10"""
        rank = []
        for ts_code in candidates:
            try:
                df = pro.hk_daily_adj(ts_code=ts_code, start_date=start, end_date=end)
                if df is None or df.empty:
                    continue
                df["trade_date"] = pd.to_datetime(df["trade_date"])
                df = df.sort_values("trade_date").reset_index(drop=True)
                missing = find_missing_dividends(df, ts_code)
                if missing:
                    print(f"     ⚠️ {ts_code} adj_factor缺失 {len(missing)} 笔: "
                          + ", ".join(f"{d}(HKD{a:.4f})" for d, a in missing))
                    df = apply_missing_adj(df, missing)
                else:
                    df['adj_factor_corrected'] = df['adj_factor']
                df = hk_compute_adjusted_prices(df)
                avg_amt = df["amount"].mean()
                rank.append((ts_code, avg_amt, df))
            except Exception as e:
                pass
        rank.sort(key=lambda x: x[1], reverse=True)
        return rank[:10]

# ---- else 块结束，以下是共享函数 ----

# ============================================================
# 唐奇安指标计算
# ============================================================
def compute_donchian_indicators(
    df: pd.DataFrame,
    entry_period: int = ENTRY_PERIOD,
    exit_period: int = EXIT_PERIOD,
    atr_period: int = ATR_PERIOD,
) -> pd.DataFrame:
    """计算唐奇安通道与 ATR 指标"""
    df = df.copy()
    df = df.sort_values("trade_date").reset_index(drop=True)

    df["TR"] = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            abs(df["high"] - df["close"].shift(1)),
            abs(df["low"] - df["close"].shift(1)),
        ),
    )
    df["ATR"] = df["TR"].rolling(atr_period, min_periods=atr_period).mean()
    df["HH_Entry"] = df["high"].rolling(entry_period).max().shift(1)
    df["LL_Entry"] = df["low"].rolling(entry_period).min().shift(1)
    df["LL_Exit"] = df["low"].rolling(exit_period).min().shift(1)
    return df


def add_ema_filter(df: pd.DataFrame, ema_fast: int, ema_slow: int) -> pd.DataFrame:
    """添加 EMA 趋势过滤器"""
    df = df.copy()
    if ema_fast > 0:
        df["EMA_Fast"] = df["close"].ewm(span=ema_fast, adjust=False).mean()
    if ema_slow > 0:
        df["EMA_Slow"] = df["close"].ewm(span=ema_slow, adjust=False).mean()
    return df


# ============================================================
# 加仓做多回测（对齐 engine.py 的 _run_long_with_pyramid）
# ============================================================
def backtest_donchian_pyramid(
    df: pd.DataFrame,
    ema_fast: int = 0,
    ema_slow: int = 0,
    entry_period: int = ENTRY_PERIOD,
    exit_period: int = EXIT_PERIOD,
    atr_period: int = ATR_PERIOD,
    n_value: float = N_VALUE,
    risk_percent: float = RISK_PERCENT / 100.0,
    initial_capital: float = INITIAL_CAPITAL,
    per_shares: int = PER_SHARES,
    max_pyramid: int = MAX_PYRAMID,
    skip_first_add: bool = SKIP_FIRST_ADD,
) -> tuple[list[dict], list[float]]:
    """唐奇安通道突破 + 金字塔加仓回测。返回 (trades, equity_curve)。"""
    df = compute_donchian_indicators(df, entry_period, exit_period, atr_period)
    df = add_ema_filter(df, ema_fast, ema_slow)

    trades = []
    capital = initial_capital
    equity_curve = [initial_capital]

    pos = None
    entry_price = 0.0
    entry_atr = 0.0
    stop_loss_price = 0.0
    stop_dist = 0.0
    current_shares = 0
    base_shares = 0
    max_shares = 0
    total_cost = 0.0
    entry_date = None
    more_info = []

    start_idx = max(entry_period, atr_period, max(ema_slow, 1)) + 2

    for i in range(start_idx, len(df)):
        row = df.iloc[i]
        current_date = row["trade_date"]

        if pos is None:
            # ---- 等待入场 ----
            atr = row["ATR"]
            if pd.isna(atr) or atr == 0:
                continue

            stop_dist = n_value * atr
            risk_amount = capital * risk_percent
            shares = int(risk_amount / (stop_dist * per_shares))
            if shares <= 0:
                continue
            base_shares = shares
            skip_offset = 1 if skip_first_add else 0
            max_shares = shares * (max_pyramid - skip_offset)

            ema_ok = True
            if ema_fast > 0 and ema_slow > 0:
                fast_val = row.get("EMA_Fast")
                slow_val = row.get("EMA_Slow")
                if pd.notna(fast_val) and pd.notna(slow_val):
                    ema_ok = fast_val > slow_val

            if row["close"] > row["HH_Entry"] and ema_ok:
                cost = row["close"] * shares * per_shares
                if cost > capital * 0.8:
                    continue
                pos = "long"
                entry_price = row["close"]
                entry_atr = row["ATR"]
                stop_loss_price = entry_price - stop_dist
                current_shares = shares
                total_cost = cost
                entry_date = current_date
                capital -= cost

        else:
            # ---- 检查出场 ----
            exit_price = None
            exit_reason = ""

            if row["close"] < stop_loss_price:
                exit_price = row["close"]
                exit_reason = "Hard Stop Loss (2×ATR)"
            elif row["close"] < row["LL_Exit"]:
                exit_price = row["close"]
                exit_reason = "Trend Reversal (10d Low)"

            # ---- 金字塔加仓 ----
            if not exit_reason and entry_atr > 0:
                skip_offset = 1 if skip_first_add else 0
                units_held = current_shares // base_shares + skip_offset
                if row["close"] > (entry_price + units_held * 0.5 * entry_atr):
                    if current_shares < max_shares:
                        add_price = row["close"]
                        stop_loss_price = add_price - n_value * entry_atr
                        add_cost = add_price * base_shares * per_shares
                        if add_cost <= capital * 0.8:
                            current_shares += base_shares
                            total_cost += add_cost
                            capital -= add_cost
                            more_info.append({
                                "add_price": add_price,
                                "add_date": current_date.strftime("%Y%m%d"),
                            })

            # ---- 出场处理 ----
            if exit_reason:
                sell_proceeds = exit_price * current_shares * per_shares
                pnl = sell_proceeds - total_cost
                capital += sell_proceeds

                trades.append({
                    "entry_date": entry_date,
                    "exit_date": current_date,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "shares": current_shares,
                    "cost": total_cost,
                    "pnl": pnl,
                    "pnl_pct": pnl / total_cost if total_cost else 0,
                    "reason": exit_reason,
                    "adds": len(more_info),
                })
                equity_curve.append(capital)

                pos = None
                entry_price = 0.0
                entry_atr = 0.0
                stop_loss_price = 0.0
                current_shares = 0
                base_shares = 0
                max_shares = 0
                total_cost = 0.0
                entry_date = None
                more_info = []

    # ---- 强制平仓 ----
    if pos is not None:
        last_price = df.iloc[-1]["close"]
        last_date = df.iloc[-1]["trade_date"]
        sell_proceeds = last_price * current_shares * per_shares
        pnl = sell_proceeds - total_cost
        capital += sell_proceeds

        trades.append({
            "entry_date": entry_date,
            "exit_date": last_date,
            "entry_price": entry_price,
            "exit_price": last_price,
            "shares": current_shares,
            "cost": total_cost,
            "pnl": pnl,
            "pnl_pct": pnl / total_cost if total_cost else 0,
            "reason": "强制平仓",
            "adds": len(more_info),
        })
        equity_curve.append(capital)

    return trades, equity_curve


# ============================================================
# 交易评估
# ============================================================
def evaluate(trades: list[dict], equity_curve: list[float],
             initial_capital: float = INITIAL_CAPITAL) -> dict:
    """评估交易结果，返回关键指标"""
    if not trades:
        return {
            "total_trades": 0, "win_rate": 0, "total_pnl": 0,
            "total_return_pct": 0, "avg_win": 0, "avg_loss": 0,
            "profit_factor": 0, "max_drawdown_pct": 0,
            "avg_trade_pnl": 0, "final_capital": initial_capital,
            "total_adds": 0,
        }

    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]

    total_pnl = sum(t["pnl"] for t in trades)
    win_rate = len(wins) / len(trades) if trades else 0
    avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0
    avg_loss = np.mean([t["pnl"] for t in losses]) if losses else 0
    total_wins_sum = sum(t["pnl"] for t in wins) if wins else 0
    total_losses_sum = abs(sum(t["pnl"] for t in losses)) if losses else 0
    profit_factor = (total_wins_sum / total_losses_sum) if total_losses_sum > 0 else (
        float("inf") if total_wins_sum > 0 else 0)
    total_adds = sum(t.get("adds", 0) for t in trades)

    # 最大回撤
    peak = equity_curve[0]
    max_dd = 0.0
    for v in equity_curve:
        if v > peak:
            peak = v
        dd = (peak - v) / peak
        if dd > max_dd:
            max_dd = dd

    final_capital = equity_curve[-1]

    return {
        "total_trades": len(trades),
        "win_rate": round(win_rate * 100, 1),
        "total_pnl": round(total_pnl, 2),
        "total_return_pct": round((final_capital / initial_capital - 1) * 100, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "avg_trade_pnl": round(total_pnl / len(trades), 2) if trades else 0,
        "final_capital": round(final_capital, 2),
        "total_adds": total_adds,
    }




# ============================================================
# 参数扫描 & 综合评分
# ============================================================
def run_parameter_scan(top10_stocks) -> pd.DataFrame:
    """对 TOP10 股票扫描所有 EMA 参数组合，返回综合评分 DataFrame"""
    all_results = {}  # ts_code -> [(fast, slow, metrics), ...]

    for rank, (ts_code, avg_amt, df_raw) in enumerate(top10_stocks, 1):
        df = df_raw.sort_values("trade_date").reset_index(drop=True)

        results = []

        # 无过滤器基准
        trades, eq = backtest_donchian_pyramid(df, ema_fast=0, ema_slow=0)
        baseline = evaluate(trades, eq)
        baseline["ema_fast"] = 0
        baseline["ema_slow"] = 0
        results.append((0, 0, baseline))

        # 参数扫描
        for fast, slow in product(FAST_PERIODS, SLOW_PERIODS):
            if slow <= fast:
                continue
            trades, eq = backtest_donchian_pyramid(df, ema_fast=fast, ema_slow=slow)
            metrics = evaluate(trades, eq)
            metrics["ema_fast"] = fast
            metrics["ema_slow"] = slow
            results.append((fast, slow, metrics))

        all_results[ts_code] = results

        best = max(results, key=lambda x: x[2]["total_return_pct"])
        print(f"#{rank} {ts_code} (日均成交 {avg_amt:,.0f})")
        print(f"  基准(无过滤):  回报 {baseline['total_return_pct']:>7.1f}%, "
              f"胜率 {baseline['win_rate']:>5.1f}%, 交易 {baseline['total_trades']:>3}次, "
              f"加仓 {baseline['total_adds']:>2}次, 回撤 {baseline['max_drawdown_pct']:>5.1f}%, "
              f"盈亏比 {baseline['profit_factor']:>5.2f}")
        print(f"  最佳(总回报): EMA({best[0]}/{best[1]}) → "
              f"回报 {best[2]['total_return_pct']:>7.1f}%, "
              f"胜率 {best[2]['win_rate']:>5.1f}%, 交易 {best[2]['total_trades']:>3}次, "
              f"加仓 {best[2]['total_adds']:>2}次, 回撤 {best[2]['max_drawdown_pct']:>5.1f}%, "
              f"盈亏比 {best[2]['profit_factor']:>5.2f}")

    # ---- 跨股票汇总 ----
    param_scores = {}
    for fast, slow in product(FAST_PERIODS, SLOW_PERIODS):
        if slow <= fast:
            continue
        param_scores[(fast, slow)] = {
            "returns": [], "win_rates": [], "profit_factors": [],
            "max_dds": [], "trade_counts": [], "add_counts": [],
            "return_vs_baseline": [], "dd_vs_baseline": [],
        }
    param_scores[(0, 0)] = {
        "returns": [], "win_rates": [], "profit_factors": [],
        "max_dds": [], "trade_counts": [], "add_counts": [],
        "return_vs_baseline": [], "dd_vs_baseline": [],
    }

    for ts_code, results in all_results.items():
        baseline_return = None
        baseline_dd = None
        for fast, slow, metrics in results:
            if fast == 0 and slow == 0:
                baseline_return = metrics["total_return_pct"]
                baseline_dd = metrics["max_drawdown_pct"]
                param_scores[(0, 0)]["returns"].append(metrics["total_return_pct"])
                param_scores[(0, 0)]["win_rates"].append(metrics["win_rate"])
                param_scores[(0, 0)]["profit_factors"].append(metrics["profit_factor"])
                param_scores[(0, 0)]["max_dds"].append(metrics["max_drawdown_pct"])
                param_scores[(0, 0)]["trade_counts"].append(metrics["total_trades"])
                param_scores[(0, 0)]["add_counts"].append(metrics["total_adds"])
            else:
                param_scores[(fast, slow)]["returns"].append(metrics["total_return_pct"])
                param_scores[(fast, slow)]["win_rates"].append(metrics["win_rate"])
                param_scores[(fast, slow)]["profit_factors"].append(metrics["profit_factor"])
                param_scores[(fast, slow)]["max_dds"].append(metrics["max_drawdown_pct"])
                param_scores[(fast, slow)]["trade_counts"].append(metrics["total_trades"])
                param_scores[(fast, slow)]["add_counts"].append(metrics["total_adds"])
                if baseline_return is not None:
                    param_scores[(fast, slow)]["return_vs_baseline"].append(
                        metrics["total_return_pct"] - baseline_return)
                    param_scores[(fast, slow)]["dd_vs_baseline"].append(
                        baseline_dd - metrics["max_drawdown_pct"])

    # ---- 构建汇总表 ----
    summary = []
    for (fast, slow), scores in param_scores.items():
        summary.append({
            "ema_fast": fast,
            "ema_slow": slow,
            "avg_return": round(np.mean(scores["returns"]), 2) if scores["returns"] else 0,
            "avg_win_rate": round(np.mean(scores["win_rates"]), 1) if scores["win_rates"] else 0,
            "avg_profit_factor": round(np.mean(scores["profit_factors"]), 2) if scores["profit_factors"] else 0,
            "avg_max_dd": round(np.mean(scores["max_dds"]), 2) if scores["max_dds"] else 0,
            "avg_trades": round(np.mean(scores["trade_counts"]), 1) if scores["trade_counts"] else 0,
            "avg_adds": round(np.mean(scores["add_counts"]), 1) if scores["add_counts"] else 0,
            "avg_return_improve": round(np.mean(scores["return_vs_baseline"]), 2) if scores["return_vs_baseline"] else 0,
            "avg_dd_improve": round(np.mean(scores["dd_vs_baseline"]), 2) if scores["dd_vs_baseline"] else 0,
        })

    summary_df = pd.DataFrame(summary)

    # ---- 综合评分 ----
    def _norm(s):
        mn, mx = s.min(), s.max()
        if mx == mn:
            return pd.Series(0.5, index=s.index)
        return (s - mn) / (mx - mn)

    summary_df["score_return"] = _norm(summary_df["avg_return"])
    summary_df["score_pf"] = _norm(summary_df["avg_profit_factor"])
    summary_df["score_wr"] = _norm(summary_df["avg_win_rate"])
    summary_df["score_dd"] = _norm(1 - summary_df["avg_max_dd"] / 100)
    summary_df["score_trades"] = _norm(summary_df["avg_trades"])
    summary_df["score_return_improve"] = _norm(summary_df["avg_return_improve"])

    # 权重：回报改善35% + 盈亏比20% + 绝对回报15% + 回撤控制15% + 胜率10% + 交易5%
    summary_df["composite_score"] = (
        summary_df["score_return_improve"] * 0.35 +
        summary_df["score_pf"] * 0.20 +
        summary_df["score_return"] * 0.15 +
        summary_df["score_dd"] * 0.15 +
        summary_df["score_wr"] * 0.10 +
        summary_df["score_trades"] * 0.05
    )

    summary_df = summary_df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    summary_df["rank"] = range(1, len(summary_df) + 1)
    return summary_df


# ============================================================
# 主流程
# ============================================================
def main():
    token = os.environ.get("TUSHARE_TOKEN")
    if not token:
        print("❌ 请先设置环境变量 TUSHARE_TOKEN")
        return
    pro = ts.pro_api(token)

    # ---- 1. 筛选流动性 TOP10 ----
    currency = "CNY" if MARKET == "A" else "HKD"
    market_label = "A股" if MARKET == "A" else "港股"
    print("=" * 80)
    print(f"🏆 近五年{market_label}日均成交额 TOP10")
    print("=" * 80)

    if MARKET == "A":
        # A股蓝筹候选池 (沪深300核心 + 行业龙头)
        candidates = A_CANDIDATES
        print(f"   候选池: A股蓝筹 {len(candidates)} 只")
    else:
        candidates = HK_CANDIDATES

    top10 = fetch_top10_by_liquidity(pro, candidates, START_DATE, END_DATE)

    if MARKET == "A":
        sb = pro.stock_basic(list_status='L', fields='ts_code,name')
        name_map = dict(zip(sb['ts_code'], sb['name']))
    else:
        hk_basic = pro.hk_basic()
        name_map = {}
        for _, r in hk_basic.iterrows():
            name_map[r["ts_code"]] = r["name"]

    for i, (code, amt, _) in enumerate(top10, 1):
        name = name_map.get(code, "")
        print(f"  #{i} {code} {name}  日均成交额: {amt:>15,.0f} {currency}")

    # ---- 2. EMA 参数扫描 ----
    print(f"\n{'=' * 80}")
    print(f"📊 EMA 趋势过滤器参数扫描 ({market_label} 加仓做多)")
    print(f"   快线: {FAST_PERIODS}")
    print(f"   慢线: {SLOW_PERIODS}")
    print(f"   唐奇安通道: entry={ENTRY_PERIOD}, exit={EXIT_PERIOD}, atr={ATR_PERIOD}")
    print(f"   加仓: max_pyramid={MAX_PYRAMID}, skip_first_add={SKIP_FIRST_ADD}")
    print(f"{'=' * 80}\n")
    summary = run_parameter_scan(top10)

    # ---- 3. 输出排名 ----
    print(f"\n{'=' * 80}")
    print("🏆 综合排名 (评分权重: 回报改善35% + 盈亏比20% + 绝对回报15% + 回撤控制15% + 胜率10% + 交易5%)")
    print(f"{'=' * 80}")
    cols = ["rank", "ema_fast", "ema_slow", "avg_return", "avg_return_improve",
            "avg_win_rate", "avg_profit_factor", "avg_max_dd", "avg_trades",
            "avg_adds", "composite_score"]
    print(summary[cols].head(25).to_string(index=False))

    # ---- 4. 分组统计 ----
    print(f"\n{'=' * 80}")
    print("📊 按慢线分组")
    print(f"{'=' * 80}")
    for slow in [0] + SLOW_PERIODS:
        sub = summary[summary["ema_slow"] == slow]
        if sub.empty:
            continue
        top3_fast = sub.head(3)["ema_fast"].tolist()
        label = "无过滤器" if slow == 0 else f"EMA_Slow={slow}"
        print(f"  {label:<18} 组合:{len(sub):>2}  均分:{sub['composite_score'].mean():.4f}  "
              f"均回报改善:{sub['avg_return_improve'].mean():>+6.1f}%  Top快线:{top3_fast}")

    print(f"\n{'=' * 80}")
    print("📊 按快线分组")
    print(f"{'=' * 80}")
    for fast in [0] + FAST_PERIODS:
        sub = summary[summary["ema_fast"] == fast]
        if sub.empty:
            continue
        top3_slow = sub.head(3)["ema_slow"].tolist()
        label = "无过滤器" if fast == 0 else f"EMA_Fast={fast}"
        print(f"  {label:<18} 组合:{len(sub):>2}  均分:{sub['composite_score'].mean():.4f}  "
              f"均回报改善:{sub['avg_return_improve'].mean():>+6.1f}%  Top慢线:{top3_slow}")

    # ---- 5. 最优建议 ----
    best = summary[summary["ema_fast"] > 0].iloc[0]
    print(f"\n{'=' * 80}")
    print("🎯 最优参数建议")
    print(f"{'=' * 80}")
    print(f"  推荐 EMA_Fast = {best['ema_fast']}, EMA_Slow = {best['ema_slow']}")
    print(f"  综合评分: {best['composite_score']:.4f}")
    print(f"  平均回报: {best['avg_return']}% (相对基准改善 {best['avg_return_improve']:+.1f}%)")
    print(f"  平均胜率: {best['avg_win_rate']}%")
    print(f"  平均盈亏比: {best['avg_profit_factor']}")
    print(f"  平均回撤: {best['avg_max_dd']}%")
    print(f"  平均交易次数: {best['avg_trades']}")

    # ---- 6. 保存 ----
    summary.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ 结果已保存至 {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
