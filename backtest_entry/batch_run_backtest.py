#!/usr/bin/env python3
"""批量回测 TOP10 流动性 A股/港股 —— 通道策略适配度排名

功能：
  1. 筛选 A 股 / 港股近一年日均成交额>1亿的所有股票
  2.1. 扫描全市场最优通道参数（唐奇安 EMA 或 布林线 period/k）;使用 core 管线逐只运行海龟策略回测（加仓做多）,返回参数组合，综合评分
  2.2. 单只扫描最优通道参数,使用 core 管线运行海龟策略回测（加仓做多）,返回参数组合，综合评分
  3. 输出全市场股票综合评分排名，推荐对应策略最佳10只股票
  备注: 按四项指标综合评分排名：
     - 期望收益率 (20%)
     - 回归年度回报率 (25%)
     - R立方 (30%)
     - 稳健夏普比率 (25%)

用法：
  python batch_run_backtest.py

切换通道：修改脚本顶部 CHANNEL_TYPE = "donchian" | "bollinger"

依赖：
  tushare, akshare (港股分红补权), pandas, numpy
  环境变量 TUSHARE_TOKEN 需要已配置
"""

import os
import sys
import importlib
import warnings
import pandas as pd
import numpy as np
import tushare as ts
from itertools import product

warnings.filterwarnings("ignore")

from core.config import BacktestConfig
from core.data import fetch_data
from core.indicators import compute_indicators
from core.engine import run_backtest
from core.statistics import calc_stats

# ============================================================
# 配置
# ============================================================
# 策略参数（与 run_backtest.py 默认一致）
CHANNEL_TYPE = "donchian"  # "donchian" | "bollinger" — 切换通道类型
MODE = "long_more"
SCAN_ALL = False #是否扫描全市场最优参数
MAX_PYRAMID = 5
ENTRY_PERIOD = 20
EXIT_PERIOD = 10
ATR_PERIOD = 14
N_VALUE = 2.0
RISK_PERCENT = 1.0
INITIAL_CAPITAL = 200000.0
YEARS = 8
FAST_PERIODS = [20, 25, 30, 40, 50]
SLOW_PERIODS = [50, 60, 80, 100, 150, 200, 250, 300, 350]

BB_PERIODS = [10, 15, 20, 25, 30]
BB_K_VALUES = [1.0, 1.5, 2.0, 2.5, 3.0]

# ============================================================
# 工具函数
# ============================================================
def _get_name_map(pro, market: str) -> dict[str, str]:
    """获取 ts_code → 股票名称 映射"""
    if market == "A":
        df = pro.stock_basic(list_status="L", fields="ts_code,name")
        return dict(zip(df["ts_code"], df["name"]))
    else:
        df = pro.hk_basic()
        name_map = {}
        for _, r in df.iterrows():
            name_map[r["ts_code"]] = r["name"]
        return name_map


def load_ode_module(market: str):
    """动态加载/重载 optimize_donchian_ema 模块（切换 A/HK 版本）"""
    os.environ["TURTLE_MARKET"] = market
    mod_name = "optimize_donchian_ema"
    if mod_name in sys.modules:
        importlib.reload(sys.modules[mod_name])
    import optimize_donchian_ema as ode
    return ode


def get_stocks(pro, market: str) -> list[tuple[str, float, object]]:
    """获取指定市场日均成交额>1亿的股票，返回 [(ts_code, avg_amt, df_raw), ...]"""
    ode = load_ode_module(market)
    return ode.fetch_liquidity(pro, ode.START_DATE, ode.END_DATE)


def get_best_ema_params(market: str, liquidity_up_to_1y: list, name_map) -> tuple[int, int]:
    """扫描 EMA 参数，返回最优 (ema_fast, ema_slow)

    注意：调用前须已通过 get_top10_stocks 加载对应 market 的模块。
    """
    ode = sys.modules["optimize_donchian_ema"]
    label = "A股" if market == "A" else "港股"
    print(f"\n📊 扫描{label}最优 EMA 趋势过滤器参数 ...")
    summary = ode.run_parameter_scan(liquidity_up_to_1y,  name_map)
    best = summary[summary["ema_fast"] > 0].iloc[0]
    ema_fast = int(best["ema_fast"])
    ema_slow = int(best["ema_slow"])
    print(f"  ✅ {label}最优 EMA: Fast={ema_fast}, Slow={ema_slow}  (综合评分: {best['composite_score']:.4f})")
    return ema_fast, ema_slow


def get_best_bb_params(pro, market: str, liquidity_up_to_1y: list) -> tuple[int, float]:
    """扫描布林线参数，返回最优 (bb_period, bb_k)"""
    # 动态加载 optimize_bollinger 模块
    mod_name = "optimize_bollinger"
    if mod_name in sys.modules:
        importlib.reload(sys.modules[mod_name])
    import optimize_bollinger as ob

    label = "A股" if market == "A" else "港股"
    print(f"\n📊 扫描{label}最优布林线参数 (bb_period, bb_k) ...")
    summary = ob.run_bb_parameter_scan(pro, liquidity_up_to_1y)
    best = summary.iloc[0]
    bb_period = int(best["bb_period"])
    bb_k = float(best["bb_k"])
    print(f"  ✅ {label}最优布林线: period={bb_period}, k={bb_k}  (综合得分: {best['综合得分']:.4f})")
    return bb_period, bb_k


def  run_single_scan_backtest(df_row,config: BacktestConfig) -> tuple[dict, list, pd.DataFrame] | None:
    """执行单只股票回测，返回 (stats_dict, trades, df)；失败返回 None"""
    try:
        if CHANNEL_TYPE == "bollinger":
            if config.bb_period==0 and config.bb_k==0:
                x_y_list = product(BB_PERIODS, BB_K_VALUES)
            else:
                x_y_list = product([config.bb_period], [config.bb_k])
        else:
            if config.ema_fast==0 and config.ema_slow==0:
                x_y_list = product(FAST_PERIODS, SLOW_PERIODS)
            else:
                x_y_list = product([config.ema_fast], [config.ema_slow])
        results = []
        for x, y in x_y_list:
            if y == x:
                continue
            config.ema_fast = x
            config.ema_slow = y
            config.bb_period = x
            config.bb_k = y
            df = compute_indicators(
                df_row,
                entry_period=config.entry_period,
                exit_period=config.exit_period,
                atr_period=config.atr_period,
                ema_fast=config.ema_fast,
                ema_slow=config.ema_slow,
                channel_type=config.channel_type,
                bb_period=config.bb_period,
                bb_k=config.bb_k,
            )
            print(f"=========={x}_{y}_{config.ts_code} 开始生成traders===========")
            trades, equity = run_backtest(df, config)
            if len(trades)<21:
                continue
            stats = calc_stats(trades, config.initial_capital, equity)
            if (stats['胜率 (%)']<25 or stats['胜率 (%)']>38) or stats['期望收益率']<1:
                continue
            if stats['回归年度回报率 (%)']<=2.5 or stats['R立方']<=0.8 or stats['稳健夏普比率']<=0.3:
                continue
            stats['ema_fast']=config.ema_fast
            stats['ema_slow']=config.ema_slow
            stats['bb_period']=config.bb_period
            stats['bb_k']=config.bb_k
            results.append({'stats': stats, 'trades': trades, 'df': df})

            # ———— 3. 综合评分排名 ————
        if not results:
            print("❌ 没有成功获取任何回测结果")
            return None

        stats_list = [r['stats'] for r in results]
        stats_rank = compute_ranking(stats_list, config.ts_code)
        best = stats_rank[stats_rank['排名'] == 1]
        if best.empty:
            return None
        best_idx = best.index[0]
        best_stats = best.iloc[0].to_dict()
        best_trades = results[best_idx]['trades']
        best_df = results[best_idx]['df']
        return best_stats, best_trades, best_df
    except Exception as e:
        print(f"    ❌ 回测失败: {e}")
        return None


# ============================================================
# 综合评分
# ============================================================
def _norm(series: pd.Series) -> pd.Series:
    """min-max 归一化到 [0, 1]"""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(0.5, index=series.index)
    return (series - mn) / (mx - mn)


def compute_ranking(results: list[dict],ts_code) -> pd.DataFrame:
    """对回测结果计算综合评分并排名"""
    df = pd.DataFrame(results)

    # 四项核心指标归一化
    df["score_期望收益率"] = _norm(df["期望收益率"])
    df["score_回归年度回报率"] = _norm(df["回归年度回报率 (%)"])
    df["score_R立方"] = _norm(df["R立方"])
    df["score_稳健夏普比率"] = _norm(df["稳健夏普比率"])

    # 加权综合得分
    df["综合得分"] = (
        df["score_期望收益率"] * 0.20 +
        df["score_回归年度回报率"] * 0.25 +
        df["score_R立方"] * 0.30 +
        df["score_稳健夏普比率"] * 0.25
    )

    df = df.sort_values("综合得分", ascending=False).reset_index(drop=True)
    df["排名"] = range(1, len(df) + 1)

    # ————  保存 CSV ————
    # OUTPUT_CSV = os.path.join(os.path.dirname(__file__), f"{ts_code}_xy_param_ranking.csv")
    # df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    # print(f"\n✅ 排名结果已保存至 {OUTPUT_CSV}")

    return df


# ============================================================
# 跨股票综合评分（重新归一化，评价范围不同）
# ============================================================
def compute_cross_stock_ranking(all_stats: list[dict]) -> pd.DataFrame:
    """对全市场股票重新计算综合评分并排名（跨股票归一化）"""
    df = pd.DataFrame(all_stats)

    df["score_期望收益率"] = _norm(df["期望收益率"])
    df["score_回归年度回报率"] = _norm(df["回归年度回报率 (%)"])
    df["score_R立方"] = _norm(df["R立方"])
    df["score_稳健夏普比率"] = _norm(df["稳健夏普比率"])

    df["综合得分"] = (
        df["score_期望收益率"] * 0.20 +
        df["score_回归年度回报率"] * 0.25 +
        df["score_R立方"] * 0.30 +
        df["score_稳健夏普比率"] * 0.25
    )

    df = df.sort_values("综合得分", ascending=False).reset_index(drop=True)
    df["排名"] = range(1, len(df) + 1)
    return df


# ============================================================
# 信号状态判断
# ============================================================
def determine_signal_state(best_trades: list[dict], best_df: pd.DataFrame,
                           channel_type: str) -> dict:
    """判断一只股票当前的信号状态

    返回字段：
        state:          'entry' 等待入场 | 'exit' 等待出场
        entry_price:    通道入场价（最后一根K线）
        exit_price:     通道出场价（最后一根K线）
        current_price:  当前收盘价
        distance_pct:   当前价距离目标价的百分比
        last_reason:    最后一笔交易的出场原因
    """
    last_row = best_df.iloc[-1]
    current_price = last_row["Close"]

    # 通道入场 / 出场价格
    if channel_type == "bollinger":
        entry_price = last_row.get("BB_Upper", current_price)
        exit_price = last_row.get("BB_Mid", current_price)
    else:
        entry_price = last_row.get("HH_Entry", current_price)
        exit_price = last_row.get("LL_Exit", current_price)

    if not best_trades or len(best_trades) == 0:
        return {
            "state": "entry",
            "entry_price": entry_price,
            "exit_price": 0,
            "current_price": current_price,
            "distance_pct": (entry_price - current_price) / current_price * 100 if entry_price else 0,
            "last_reason": "",
        }

    last_trade = best_trades[-1]
    last_reason = last_trade.get("Reason", "")

    # 强制平仓（end-of-data 强制清算）不算已出场 → 仍视为持仓中
    if last_reason == "强制平仓":
        distance_pct = (current_price - exit_price) / current_price * 100 if exit_price else 0
        return {
            "state": "exit",
            "entry_price": 0,
            "exit_price": exit_price,
            "current_price": current_price,
            "distance_pct": distance_pct,
            "last_reason": last_reason,
        }

    # 正常平仓（趋势出场 / 硬止损）→ 等待入场信号
    distance_pct = (entry_price - current_price) / current_price * 100 if entry_price else 0
    return {
        "state": "entry",
        "entry_price": entry_price,
        "exit_price": exit_price,
        "current_price": current_price,
        "distance_pct": distance_pct,
        "last_reason": last_reason,
    }


# ============================================================
# 保存单只股票结果到 Excel
# ============================================================
def save_single_stock_excel(stats: dict, trades: list[dict], df: pd.DataFrame,
                            output_dir: str, ts_code: str):
    """将单只股票的 stats / trades / df 保存到同一个 Excel 文件"""
    filepath = os.path.join(output_dir, f"{ts_code}_backtest_result.xlsx")
    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        # Sheet 1: 统计指标
        pd.DataFrame([stats]).to_excel(writer, sheet_name="stats", index=False)
        # Sheet 2: 交易记录
        if trades:
            pd.DataFrame(trades).to_excel(writer, sheet_name="trades", index=False)
        else:
            pd.DataFrame({"提示": ["无交易记录"]}).to_excel(writer, sheet_name="trades", index=False)
        # Sheet 3: 指标数据
        df.to_excel(writer, sheet_name="df", index=True)
    print(f"    📁 已保存: {filepath}")


# ============================================================
# 输出
# ============================================================
def print_results(df: pd.DataFrame):
    """打印排名结果"""
    # 分市场打印
    for market in ("A", "HK"):
        sub = df[df["market"] == market]
        label = "A股" if market == "A" else "港股"
        print(f"\n{'=' * 130}")
        print(f"🏆 {label} TOP10 {CHANNEL_TYPE}策略适配度排名")
        print(f"{'=' * 130}")

        cols = [
            "排名", "ts_code", "name", "综合得分",
            "期望收益率", "回归年度回报率 (%)", "R立方", "稳健夏普比率",
            "总收益率 (%)", "胜率 (%)", "最大回撤 (%)", "CAGR (%)", "总交易",
        ]
        # 使用全市场排名
        print(sub[cols].to_string(index=False))

    # 全市场综合排名
    print(f"\n{'=' * 130}")
    print("🌏 全市场综合排名 (A股 + 港股)")
    print(f"{'=' * 130}")
    cols_full = [
        "排名", "market", "ts_code", "name", "综合得分",
        "期望收益率", "回归年度回报率 (%)", "R立方", "稳健夏普比率",
        "总收益率 (%)", "胜率 (%)", "最大回撤 (%)",
    ]
    print(df[cols_full].to_string(index=False))

    # 最佳推荐
    print(f"\n{'=' * 80}")
    print(f"🎯 最佳推荐 — 两个市场最适合{CHANNEL_TYPE}策略的股票")
    print(f"{'=' * 80}")

    for market in ("A", "HK"):
        best = df[df["market"] == market].iloc[0]
        label = "A股" if market == "A" else "港股"
        print(f"\n  {label} 最佳: {best['ts_code']} {best['name']}")
        print(f"    综合得分: {best['综合得分']:.4f}")
        print(f"    期望收益率: {best['期望收益率']:.4f}  |  "
              f"回归年度回报率: {best['回归年度回报率 (%)']:.2f}%  |  "
              f"R立方: {best['R立方']:.2f}  |  "
              f"稳健夏普: {best['稳健夏普比率']:.2f}")
        print(f"    总收益率: {best['总收益率 (%)']:.2f}%  |  "
              f"胜率: {best['胜率 (%)']}%  |  "
              f"最大回撤: {best['最大回撤 (%)']}%  |  "
              f"CAGR: {best['CAGR (%)']}%  |  "
              f"交易: {best['总交易']}笔")


# ============================================================
# 主流程
# ============================================================
def main():
    token = os.environ.get("TUSHARE_TOKEN")
    if not token:
        print("❌ 请先设置环境变量 TUSHARE_TOKEN")
        return
    pro = ts.pro_api(token)



    for market in ("A","HK"):
        label = "A股" if market == "A" else "港股"
        currency = "CNY" if market == "A" else "HKD"
        output_dir = os.path.join(os.path.dirname(__file__),
                                  f"{CHANNEL_TYPE}_{market}_{MODE}_result")
        os.makedirs(output_dir, exist_ok=True)

        # ———— 1. 获取流动性 TOP10 ————
        print(f"\n{'=' * 80}")
        print(f"📊 筛选{label}近一年日均成交额 TOP10 ...")
        print(f"{'=' * 80}")
        liquidity_up_to_1y = get_stocks(pro, market)

        name_map = _get_name_map(pro, market)
        for i, (ts_code, avg_amt, _) in enumerate(liquidity_up_to_1y, 1):
            name = name_map.get(ts_code, "")
            print(f"  #{i} {ts_code} {name}  日均成交额: {avg_amt:>15,.0f} {currency}")

        if SCAN_ALL:
            # ———— 2. 扫描全市场最优参数 ————
            if CHANNEL_TYPE == "bollinger":
                bb_period, bb_k = get_best_bb_params(pro, market, liquidity_up_to_1y)
                ema_fast, ema_slow = 0,0
            else:
                ema_fast, ema_slow = get_best_ema_params(market, liquidity_up_to_1y, name_map)
                bb_period, bb_k = 0,0
        else:
            bb_period, bb_k = 0, 0
            ema_fast, ema_slow = 0,0

        # ———— 3. 逐只回测 ————
        all_results = []
        for i, (ts_code, avg_amt, df_row) in enumerate(liquidity_up_to_1y, 1):
            name = name_map.get(ts_code, "")
            print(f"\n[{i}/10] {ts_code} {name} ...")

            config = BacktestConfig(
                ts_code=ts_code,
                stock_name=name,
                market=market,
                years=YEARS,
                mode=MODE,
                max_pyramid=MAX_PYRAMID,
                entry_period=ENTRY_PERIOD,
                exit_period=EXIT_PERIOD,
                atr_period=ATR_PERIOD,
                n_value=N_VALUE,
                risk_percent=RISK_PERCENT,
                initial_capital=INITIAL_CAPITAL,
                channel_type=CHANNEL_TYPE,
                ema_fast=ema_fast,
                ema_slow=ema_slow,
                bb_period=bb_period,
                bb_k=bb_k,
            )

            result = run_single_scan_backtest(df_row,config)
            if result is None:
                continue
            stats, best_trades, best_df = result

            stats["market"] = market
            stats["ts_code"] = ts_code
            stats["name"] = name
            stats["avg_amt"] = avg_amt
            all_results.append({
                "stats": stats,
                "trades": best_trades,
                "df": best_df,
            })

            # 保存单只股票 Excel（stats / trades / df 三个 sheet）
            save_single_stock_excel(stats, best_trades, best_df, output_dir, ts_code)

            print(f"    ✅ 回归年度回报率 (%): {stats['回归年度回报率 (%)']}%  |  "
                  f"胜率: {stats['胜率 (%)']}%  |  "
                  f"回撤: {stats['最大回撤 (%)']}%  |  "
                  f"交易: {stats['总交易']}笔  |  "
                  f"R立方: {stats['R立方']}  |  "
                  f"稳健夏普: {stats['稳健夏普比率']}")

        # ———— 3. 跨股票综合评分排名 ————
        if not all_results:
            print("❌ 没有成功获取任何回测结果")
            return

        stats_list = [r["stats"] for r in all_results]
        ranking_df = compute_cross_stock_ranking(stats_list)

        # ———— 4. 信号距离写入 stats, 按入场/出场排序保存 Excel ————
        for r in all_results:
            sig = determine_signal_state(r["trades"], r["df"], CHANNEL_TYPE)
            r["stats"]["signal_state"] = sig["state"]
            r["stats"]["entry_price"] = sig["entry_price"]
            r["stats"]["exit_price"] = sig["exit_price"]
            r["stats"]["current_price"] = sig["current_price"]
            r["stats"]["distance_pct"] = sig["distance_pct"]

        # 排序：入场信号在前，出场信号在后；组内按 distance_pct 升序
        entry_list = [r for r in all_results if r["stats"]["signal_state"] == "entry"]
        exit_list = [r for r in all_results if r["stats"]["signal_state"] == "exit"]
        entry_list.sort(key=lambda x: x["stats"]["distance_pct"])
        exit_list.sort(key=lambda x: x["stats"]["distance_pct"])
        ordered = entry_list + exit_list

        # 按排序后的顺序更新综合得分和排名（使用跨股票评分）
        score_cols = ["综合得分", "排名", "score_期望收益率", "score_回归年度回报率",
                      "score_R立方", "score_稳健夏普比率"]
        ranking_lookup = ranking_df.set_index("ts_code")[score_cols].to_dict("index")
        ordered_stats = []
        for idx, r in enumerate(ordered, 1):
            s = r["stats"].copy()
            row = ranking_lookup.get(s["ts_code"], {})
            s["综合得分"] = row.get("综合得分", 0)
            s["排名"] = row.get("排名", 0)
            s["score_期望收益率"] = row.get("score_期望收益率", 0)
            s["score_回归年度回报率"] = row.get("score_回归年度回报率", 0)
            s["score_R立方"] = row.get("score_R立方", 0)
            s["score_稳健夏普比率"] = row.get("score_稳健夏普比率", 0)
            ordered_stats.append(s)

        OUTPUT_EXCEL = os.path.join(output_dir,
                                    f"{CHANNEL_TYPE}_{market}_{MODE}_signal_ranking.xlsx")
        with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
            pd.DataFrame(ordered_stats).to_excel(writer, sheet_name="信号排名", index=False)
        print(f"\n✅ 信号排名已保存至 {OUTPUT_EXCEL}")


if __name__ == "__main__":
    main()
