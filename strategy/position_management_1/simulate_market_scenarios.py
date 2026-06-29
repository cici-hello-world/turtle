#!/usr/bin/env python3
"""模拟未来6个月多种K线形态 + 仓位管理回测

生成 8 类模拟数据：
  1. 单边上涨     — 趋势持续向上，低波动
  2. 单边下跌     — 趋势持续向下，低波动
  3. 震荡向上     — 高波动 + 正弦摆动，重心上移
  4. 震荡向下     — 高波动 + 大幅正弦摆动，重心缓慢下移（明显区别于单边下跌）
  5. 横盘震荡     — 价格在箱体内反复，重心几乎不变（网格策略最适场景）
  6. V形反转      — 先急跌再急涨，回到原点附近
  7. 倒V反转      — 先急涨再急跌，回到原点附近
  8. 缓慢阴跌     — 极缓慢下跌，夹杂小反弹（温水煮青蛙）

绘制每类的模拟K线图，然后用最优网格参数跑回测，输出资金曲线对比。
"""

import sys
import os
from datetime import timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.config import BacktestConfig
from core.indicators import compute_indicators
from core.engine import run_backtest
from core.statistics import calc_stats


# ==================== 老铺最优参数 ====================
# BEST_PARAMS = dict(
#    pm_grid_down       = 0.12,
#    pm_grid_up         = 0.03,
#    pm_reduce_pct      = 0.25,
#    pm_add_pct         = 0.01,
#    pm_adx_threshold   = 25,
#    pm_grid_down_agg   = 0.02,
#    pm_grid_up_agg     = 0.03,
#    pm_reduce_pct_agg  = 0.75,
# )
# 理想最优参数
# BEST_PARAMS = dict(
#    pm_grid_down       = 0.05,
#    pm_grid_up         = 0.03,
#    pm_reduce_pct      = 0.25,
#    pm_add_pct         = 0.01,
#    pm_adx_threshold   = 25,
#    pm_grid_down_agg   = 0.05,
#    pm_grid_up_agg     = 0.02,
#    pm_reduce_pct_agg  = 0.5,
#     pm_take_profit_ratio = 1.30,
# )
# 泡泡玛特最优参数
BEST_PARAMS = dict(
   pm_grid_down       = 0.03,
   pm_grid_up         = 0.05,
   pm_reduce_pct      = 0.25,
   pm_add_pct         = 0.01,
   pm_adx_threshold   = 35,
   pm_grid_down_agg   = 0.03,
   pm_grid_up_agg     = 0.05,
   pm_reduce_pct_agg  = 0.25,
)

# 美团最优参数
# BEST_PARAMS = dict(
#     pm_grid_down=0.07,
#     pm_grid_up = 0.05,
#     pm_reduce_pct = 0.25,
#     pm_add_pct = 0.03,
#     pm_adx_threshold = 25,
#     pm_grid_down_agg = 0.02,
#     pm_grid_up_agg = 0.05,
#     pm_reduce_pct_agg = 0.25,
#     pm_take_profit_ratio = 1.30,
# )

# ==================== 基础配置 ====================
# BASE_KWARGS = dict(
#     ts_code="06181.HK",
#     stock_name="老铺",
#     market="HK",
#     mode="position_management_1",
#     pm_start_date="20260625",
#     ema_slow = 0,
#     ema_fast = 0,
#     pm_initial_shares=3,
#     initial_capital=40000,
#     user_dmi= True,
#     pm_dynamic_position = False,
#     per_shares=100,
#     data_source="akshare",
# )
# BASE_KWARGS = dict(
#     ts_code="02015.HK",
#     stock_name="理想",
#     market="HK",
#     mode="position_management_1",
#     pm_start_date="20260625",
#     ema_slow = 0,
#     ema_fast = 0,
#     pm_initial_shares=25,
#     initial_capital=40000,
#     user_dmi= True,
#     pm_dynamic_position = False,
#     per_shares=100,
#     data_source="akshare",
# )
BASE_KWARGS = dict(
    ts_code="09992.HK",
    stock_name="泡泡玛特",
    market="HK",
    mode="position_management_1",
    pm_start_date="20260625",
    ema_slow = 0,
    ema_fast = 0,
    pm_initial_shares=2,
    initial_capital=40000,
    user_dmi= True,
    pm_dynamic_position = True,
    per_shares=200,
    data_source="akshare",
)
# BASE_KWARGS = dict(
#     ts_code="03690.HK",
#     stock_name="美团",
#     market="HK",
#     mode="position_management_1",
#     pm_start_date="20260625",
#     ema_slow = 0,
#     ema_fast = 0,
#     pm_initial_shares=2,
#     initial_capital=40000,
#     user_dmi= True,
#     pm_dynamic_position = False,
#     per_shares=100,
#     data_source="akshare",
# )
TRADING_DAYS = 126  # 约3个月交易日


def get_last_price() -> float:
    """获取理想汽车最近一个交易日的收盘价"""
    from core.cache import load as cache_load
    cached = cache_load(BASE_KWARGS.get("ts_code"), BASE_KWARGS.get("market"))
    if cached is not None and not cached.empty:
        return float(cached["Close"].iloc[-1]),cached["Date"].iloc[-1]
    raise ValueError(f"{BASE_KWARGS.get("ts_code")}未获取到缓存日线行情")


def setup_cjk_font():
    """查找并设置中文字体"""
    try:
        import matplotlib
        import matplotlib.font_manager as fm
        candidates = ["PingFang SC", "Heiti SC", "STHeiti",
                      "PingFang HK", "Arial Unicode MS", "Hiragino Sans GB"]
        for name in candidates:
            for f in fm.fontManager.ttflist:
                if f.name == name:
                    print(f"🔤 使用字体: {f.name}")
                    matplotlib.rcParams["font.family"] = [name, "sans-serif"]
                    matplotlib.rcParams["axes.unicode_minus"] = False
                    return
        print("⚠️ 未找到中文字体，使用英文标签")
    except Exception:
        print("⚠️ 字体设置失败")


def generate_ohlcv(start_date: str, days: int, start_price: float,
                   trend_type: str) -> pd.DataFrame:
    """生成模拟日线 OHLCV 数据

    每个形态用独立的 random seed，用「趋势项 + 震荡项 + 噪声」构造。
    """
    seed_map = {
        "up": 42, "down": 43,
        "volatile_up": 44, "volatile_down": 45,
        "sideways": 46, "v_reversal": 47,
        "inv_v_reversal": 48, "slow_grind_down": 49,
    }
    np.random.seed(seed_map[trend_type])

    dates = pd.bdate_range(start=start_date, periods=days)
    n = len(dates)
    t = np.arange(n)

    # ---- 各形态参数：trend(斜率), oscillation(振幅+周期), noise_std ----
    # 校准原则（63天≈3个月）：
    #   - 单边形态：高斜率 + 低噪声，趋势信号远大于噪声累积
    #   - 震荡形态：低斜率 + 中等正弦振幅 + 较高噪声，摆动可见
    #   - 噪声累积 ≈ noise_std × √63；趋势 ≈ slope × 63；确保 SNR > 2
    if trend_type == "up":
        trend_slope = 0.008        # 每天+0.8%，63天≈+65%
        osc_amp, osc_period = 0.0, 60
        noise_std = 0.012          # 累积噪声≈9.5%，SNR≈5.3
    elif trend_type == "down":
        trend_slope = -0.008       # 每天-0.8%，63天≈-40%
        osc_amp, osc_period = 0.0, 60
        noise_std = 0.012
    elif trend_type == "volatile_up":
        trend_slope = 0.003        # 每天+0.3%，63天≈+21%
        osc_amp, osc_period = 0.06, 14   # ±6%振幅，约4.5个完整周期
        noise_std = 0.016          # 累积噪声≈12.7%，SNR≈1.5（可见趋势但摆动明显）
    elif trend_type == "volatile_down":
        trend_slope = -0.002       # 每天-0.2%，63天≈-12%（缓跌）
        osc_amp, osc_period = 0.08, 12   # ±8%振幅，约5个周期 → 大摆动明显区别于单边下跌
        noise_std = 0.017          # 累积噪声≈13.5%
    elif trend_type == "sideways":
        trend_slope = 0.0          # 零趋势
        osc_amp, osc_period = 0.06, 20   # ±6%箱体，约3个周期
        noise_std = 0.014
    elif trend_type == "v_reversal":
        trend_slope = 0.0
        osc_amp, osc_period = 0.0, 60
        noise_std = 0.014          # 低噪声，让V形更干净
    elif trend_type == "inv_v_reversal":
        trend_slope = 0.0
        osc_amp, osc_period = 0.0, 60
        noise_std = 0.014
    elif trend_type == "slow_grind_down":
        trend_slope = -0.0035      # 每天-0.35%，63天≈-20%
        osc_amp, osc_period = 0.03, 10   # ±3%小反弹，约6个周期
        noise_std = 0.012          # 低噪声确保趋势方向可靠
    else:
        raise ValueError(f"未知趋势类型: {trend_type}")

    # ---- 构造对数日收益 ----
    # 线性趋势 → 日增量
    if trend_type in ("v_reversal", "inv_v_reversal"):
        # V形/倒V：用分段二次函数构造价格路径，再反推日收益
        if trend_type == "v_reversal":
            # 先跌后涨，最低点在中点
            half = n // 2
            path = np.zeros(n)
            path[:half] = -0.012 * np.arange(half)           # 前半跌，31天≈-37%
            path[half:] = -0.012 * half + 0.012 * np.arange(n - half)  # 后半涨回原点
            # 加噪声
            noise = np.random.normal(0, noise_std, n)
            daily_log_ret = np.zeros(n)
            daily_log_ret[1:] = path[1:] - path[:-1]
            daily_log_ret[0] = path[0]
            daily_log_ret += noise
        else:  # inv_v_reversal
            half = n // 2
            path = np.zeros(n)
            path[:half] = 0.012 * np.arange(half)             # 前半涨，31天≈+45%
            path[half:] = 0.012 * half - 0.012 * np.arange(n - half)  # 后半跌回原点
            noise = np.random.normal(0, noise_std, n)
            daily_log_ret = np.zeros(n)
            daily_log_ret[1:] = path[1:] - path[:-1]
            daily_log_ret[0] = path[0]
            daily_log_ret += noise
    else:
        trend = trend_slope * t
        oscillation = osc_amp * np.sin(2 * np.pi * t / osc_period)

        daily_drift = np.zeros(n)
        daily_drift[1:] = trend[1:] - trend[:-1]
        daily_drift[0] = trend[0]

        daily_osc = np.zeros(n)
        daily_osc[1:] = oscillation[1:] - oscillation[:-1]
        daily_osc[0] = oscillation[0]

        noise = np.random.normal(0, noise_std, n)
        daily_log_ret = daily_drift + daily_osc + noise

    close_series = start_price * np.exp(np.cumsum(daily_log_ret))

    # ---- 生成 OHLC ----
    opens = np.zeros(n)
    highs = np.zeros(n)
    lows = np.zeros(n)
    daily_volatility = np.abs(daily_log_ret) + 0.01

    for i in range(n):
        prev_close = start_price if i == 0 else close_series[i - 1]
        c = close_series[i]
        vol = daily_volatility[i]

        gap = np.random.normal(0, vol * 0.3)
        o = prev_close * (1 + gap)

        high_add = abs(np.random.normal(0, vol * 0.5))
        low_sub = abs(np.random.normal(0, vol * 0.5))
        h = max(o, c) + high_add
        l = min(o, c) - low_sub
        if l <= 0:
            l = min(o, c) * 0.98

        opens[i] = o
        highs[i] = h
        lows[i] = l

    avg_vol = 50000
    volumes = np.random.lognormal(np.log(avg_vol), 0.4, n).astype(int)
    amounts = (volumes * close_series * 100 / 1000).round(2)

    df = pd.DataFrame({
        "ts_code": "02015.HK",
        "Date": dates,
        "Open": opens.round(2),
        "High": highs.round(2),
        "Low": lows.round(2),
        "Close": close_series.round(2),
        "Volume": volumes,
        "Amount": amounts,
        "Cash_Dividend": 0.0,
    })
    df["Date"] = pd.to_datetime(df["Date"])
    return df.reset_index(drop=True)


def run_one_scenario(df: pd.DataFrame, config):
    """在模拟数据上跑一次回测，返回 (stats, trades, equity, df_sim_indicators)"""
    # 记录模拟起始日期（拼接缓存前）
    sim_start = df["Date"].min()
    start_price = df["Close"].iloc[0]
    end_price = df["Close"].iloc[-1]
    # ---- 拼接缓存历史数据，确保指标计算有足够的预热数据 ----
    from core.cache import load as cache_load
    cached = cache_load(config.ts_code, config.market)
    if cached is not None and not cached.empty:
        hist = cached[cached["Date"] < sim_start].copy()
        if not hist.empty:
            common_cols = df.columns.intersection(hist.columns)
            df = pd.concat([hist[common_cols], df[common_cols]], ignore_index=True)
            df = df.sort_values("Date").reset_index(drop=True)
            start_price = hist["Close"].iloc[-1]
            print(f"   📦 已拼接缓存历史 {len(hist)} 条 (截止 {hist['Date'].max().strftime('%Y-%m-%d')})")

    df = compute_indicators(
        df,
        entry_period=config.entry_period,
        exit_period=config.exit_period,
        atr_period=config.atr_period,
        ema_fast=config.ema_fast,
        ema_slow=config.ema_slow,
        channel_type=config.channel_type,
        bb_period=config.bb_period,
        bb_k=config.bb_k,
        am_ma_period=config.am_ma_period,
        mode=config.mode,
    )
    trades, equity, eq_dates = run_backtest(df, config)
    benchmark_return = round(((end_price - start_price)/start_price)*100,2)
    stats = calc_stats(trades, config.initial_capital, equity, eq_dates,benchmark_return=benchmark_return)

    # 切回模拟期间（去掉缓存历史），用于导出日线数据
    # df_sim = df[df["Date"] >= sim_start].copy()

    return stats, trades, equity, df


# ==================== 绘图 ====================

def plot_kline(df: pd.DataFrame, trades: list[dict], label: str, ax):
    """在给定的 ax 上绘制 K 线图 + 交易标记"""
    import matplotlib.dates as mdates

    dates = df["Date"].values
    opens = df["Open"].values
    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values

    # K线颜色：阳线红，阴线绿
    for i in range(len(df)):
        date_num = mdates.date2num(dates[i])
        color = "#e74c3c" if closes[i] >= opens[i] else "#2ecc71"
        o, c, h, l = opens[i], closes[i], highs[i], lows[i]
        ax.plot([date_num, date_num], [l, h], color=color, linewidth=0.8)
        body_h = abs(c - o) or (h - l) * 0.005
        ax.bar(date_num, body_h, 0.6, bottom=min(o, c),
               color=color, edgecolor=color, linewidth=0.3, alpha=0.9)

    # MA 均线
    ma20 = df["Close"].rolling(20).mean()
    ma60 = df["Close"].rolling(60).mean()
    ax.plot(dates, ma20, color="#3498db", linewidth=0.7, alpha=0.7, label="MA20")
    if len(df) >= 60:
        ax.plot(dates, ma60, color="#f39c12", linewidth=0.7, alpha=0.7, label="MA60")

    # 交易标记
    if trades:
        buy_dates = [t["Date"] for t in trades if t["Type"] == "加仓"]
        buy_prices = [t["Price"] for t in trades if t["Type"] == "加仓"]
        sell_dates = [t["Date"] for t in trades if t["Type"] == "减仓"]
        sell_prices = [t["Price"] for t in trades if t["Type"] == "减仓"]
        if buy_dates:
            ax.scatter(buy_dates, buy_prices, marker="^", s=50, color="#e74c3c",
                      edgecolors="white", linewidths=0.3, zorder=5, label=f"加仓({len(buy_dates)})")
        if sell_dates:
            ax.scatter(sell_dates, sell_prices, marker="v", s=50, color="#2ecc71",
                      edgecolors="white", linewidths=0.3, zorder=5, label=f"减仓({len(sell_dates)})")

    sp, ep = closes[0], closes[-1]
    chg = (ep / sp - 1) * 100
    ax.set_title(f"{label}  ¥{sp:.2f}→¥{ep:.2f} ({chg:+.2f}%)", fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=4))
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_ylabel("HKD", fontsize=7)
    ax.legend(fontsize=6, loc="upper left")
    ax.grid(True, alpha=0.3)


def plot_all_klines(sim_data: dict, all_results: dict, output_path: str):
    """多合一 K线图"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️ matplotlib 不可用")
        return

    labels = list(sim_data.keys())
    nrows = (len(labels) + 1) // 2
    fig, axes = plt.subplots(nrows, 2, figsize=(20, 4 * nrows))
    axes = axes.flatten()

    for ax, label in zip(axes, labels):
        df = sim_data[label]
        trades = all_results[label]["trades"]
        plot_kline(df, trades, label, ax)
    # 隐藏多余的子图
    for ax in axes[len(labels):]:
        ax.set_visible(False)

    fig.suptitle("模拟K线 — 多种市场形态   (🔴红=阳线 🟢绿=阴线  ▲加仓  ▼减仓)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"📊 K线图已保存: {output_path}")


def plot_equity_curves(results: dict, output_path: str):
    """多合一 资金曲线图"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️ matplotlib 不可用")
        return

    labels = list(results.keys())
    nrows = (len(labels) + 1) // 2
    fig, axes = plt.subplots(nrows, 2, figsize=(20, 4 * nrows))
    axes = axes.flatten()

    for ax, label in zip(axes, labels):
        equity = results[label]["equity"]
        if not equity:
            ax.set_title(label)
            continue
        ax.plot(equity, linewidth=0.8, color="#2c3e50")
        init = equity[0]
        final = equity[-1]
        ret = (final - init) / init * 100
        ax.axhline(y=init, color="gray", linestyle="--", alpha=0.4)
        ax.set_title(f"{label}  收益率={ret:+.2f}%  最终¥{final:,.0f}", fontsize=10)
        ax.set_ylabel("权益 (¥)", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
    for ax in axes[len(labels):]:
        ax.set_visible(False)

    fig.suptitle("仓位管理策略 — 多种市场形态资金曲线对比", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"📊 资金曲线图已保存: {output_path}")


# ==================== main ====================
if __name__ == "__main__":
    setup_cjk_font()

    # 1. 获取最新价格
    print(f"📥 获取{BASE_KWARGS.get("stock_name")}最新收盘价...")
    LAST_REAL_PRICE,LAST_DATE = get_last_price()
    print(f"   最新收盘价: {LAST_REAL_PRICE:.2f} HKD,最新日期:{LAST_DATE}\n")

    # 2. 8种形态定义
    SCENARIOS = [
        ("单边上涨",     "up"),
        ("单边下跌",     "down"),
        ("震荡向上",     "volatile_up"),
        ("震荡向下",     "volatile_down"),
        ("横盘震荡",     "sideways"),
        ("V形反转",      "v_reversal"),
        ("倒V反转",      "inv_v_reversal"),
        ("缓慢阴跌",     "slow_grind_down"),
    ]

    sim_data = {}
    for label, trend_type in SCENARIOS:
        print(f"🕯 生成「{label}」模拟K线 ({TRADING_DAYS}天) ...")
        start = (LAST_DATE + timedelta(days=1)).strftime("%Y%m%d")
        df = generate_ohlcv(start, TRADING_DAYS, LAST_REAL_PRICE, trend_type)
        sim_data[label] = df
        sp, ep = df["Close"].iloc[0], df["Close"].iloc[-1]
        lo, hi = df["Low"].min(), df["High"].max()

        # 计算一些描述性统计
        rets = df["Close"].pct_change().dropna()
        print(f"   价格 {sp:.2f}→{ep:.2f} ({(ep/sp-1)*100:+.2f}%)  "
              f"最低¥{lo:.2f} 最高¥{hi:.2f}  "
              f"日波动σ={rets.std()*100:.2f}%")

    # 3. 回测
    print("\n" + "=" * 70)
    print("🚀 开始回测 (grid_up=0.03 grid_down=0.03 add=0.030 reduce=0.75)")
    print("=" * 70)

    all_results = {}

    for label, _ in SCENARIOS:
        print(f"\n--- {label} ---")
        config = BacktestConfig(**{**BASE_KWARGS, **BEST_PARAMS})
        stats, trades, equity, df_sim = run_one_scenario(sim_data[label], config)
        all_results[label] = {"stats": stats, "trades": trades, "equity": equity, "df_sim": df_sim}

        print(f"  交易: {stats['总交易']}次  "
              f"加仓: {stats['盈利次数']+stats['亏损次数']}次  "
              f"收益率: {stats['总收益率 (%)']:+.2f}%  "
              f"盈亏: ¥{stats['总盈亏 (¥)']:+,.2f}  "
              f"回撤: {stats['最大回撤 (%)']:+.2f}%")
        print(f"  最终资金: ¥{stats['最终资金 (¥)']:,.2f}  "
              f"夏普: {stats['夏普比率']:.2f}  "
              f"CAGR: {stats['CAGR (%)']:.2f}%  "
              f"R³: {stats['R立方']:.2f}")

    # 4. 汇总对比
    print("\n" + "=" * 90)
    print("📊 形态汇总对比（按总收益率排序）")
    print("=" * 90)

    # 排序输出
    sorted_labels = sorted(SCENARIOS, key=lambda x: all_results[x[0]]["stats"]["总收益率 (%)"], reverse=True)
    print(f"{'形态':<10} {'价格变动':>10} {'总收益率':>10} {'超额收益':>10} {'总盈亏':>14} {'最大回撤':>10} {'交易':>6} {'夏普':>8} {'CAGR':>8}")
    print("-" * 90)
    for label, _ in sorted_labels:
        s = all_results[label]["stats"]
        df = sim_data[label]
        cache_df = df_sim[df_sim["Date"] < df["Date"].iloc[0]].copy()
        price_chg = (df["Close"].iloc[-1] / cache_df["Close"].iloc[-1] - 1) * 100
        print(f"{label:<10} {price_chg:>+9.2f}% {s['总收益率 (%)']:>+9.2f}% {s['超额收益 (%)']:>+9.2f}%"
              f"{s['总盈亏 (¥)']:>+13,.0f}  {s['最大回撤 (%)']:>+9.2f}% "
              f"{s['总交易']:>5}  {s['夏普比率']:>7.2f} {s['CAGR (%)']:>7.2f}%")

    # 5. 交易明细
    print("\n📋 交易明细汇总：")
    for label, _ in SCENARIOS:
        trades = all_results[label]["trades"]
        if not trades:
            print(f"  {label}: 无交易")
            continue
        tdf = pd.DataFrame(trades)
        buys = tdf[tdf["Type"] == "加仓"]
        sells = tdf[tdf["Type"] == "减仓"]
        reasons = tdf["Reason"].value_counts().index.tolist()
        print(f"  {label}: 加仓{len(buys)}次 减仓{len(sells)}次  "
              f"买入均价¥{buys['Price'].mean():.2f}" if len(buys) else f"  {label}: 加仓0次 减仓{len(sells)}次  ",
              end="")
        if len(sells):
            print(f"  卖出均价¥{sells['Price'].mean():.2f}")
        else:
            print()

    # 5b. 导出 Excel（与真实回测报告格式一致：Summary / 日线原始数据 / 仓位管理明细）
    ts_code = BASE_KWARGS["ts_code"]
    for label, _ in SCENARIOS:
        r = all_results[label]
        stats = r["stats"]
        trades = r["trades"]
        df_sim = r["df_sim"]

        safe_label = label.replace("/", "_")
        excel_path = f"simulate_{ts_code}_{safe_label}.xlsx"
        with pd.ExcelWriter(os.path.join(config.output_dir, excel_path), engine="openpyxl") as writer:
            # Summary
            summary_data = [
                ["形态", label],
                ["模拟天数", TRADING_DAYS],
                ["交易次数", stats["总交易"]],
                ["胜率 (%)", stats["胜率 (%)"]],
                ["赔率", stats["赔率"]],
                ["期望收益率", stats["期望收益率"]],
                ["总盈亏 (¥)", stats["总盈亏 (¥)"]],
                ["总收益率 (%)", stats["总收益率 (%)"]],
                ["超额收益 (%)", stats["超额收益 (%)"]],
                ["最大回撤 (%)", stats["最大回撤 (%)"]],
                ["最终资金 (¥)", stats["最终资金 (¥)"]],
                ["夏普比率", stats["夏普比率"]],
                ["CAGR (%)", stats["CAGR (%)"]],
                ["MAR比率", stats["MAR比率"]],
                ["衰落持续期 (天)", stats["衰落持续期 (天)"]],
                ["回归年度回报率 (%)", stats["回归年度回报率 (%)"]],
                ["R平方", stats["R平方"]],
                ["R立方", stats["R立方"]],
                ["稳健夏普比率", stats["稳健夏普比率"]],
            ]
            pd.DataFrame(summary_data, columns=["指标", "数值"]).to_excel(
                writer, sheet_name="Summary", index=False)

            # 日线原始数据（仅模拟期间，含指标）
            df_export = df_sim.copy()
            df_export["Date"] = df_export["Date"].dt.strftime("%Y-%m-%d")
            df_export.to_excel(writer, sheet_name="日线原始数据", index=False)

            # 仓位管理明细
            if trades:
                tdf = pd.DataFrame(trades)
                pm_cols = ["Date", "Type", "Price", "Shares", "Amount",
                           "Position_After", "Capital_After", "Equity_After",
                           "Reference_Price", "Reason"]
                tdf[pm_cols].to_excel(writer, sheet_name="仓位管理明细", index=False)

        print(f"📁 {label}: {excel_path}")

    # 6. 绘图
    plot_all_klines(sim_data, all_results, f"simulate_market_klines_{BASE_KWARGS.get("ts_code")}.png")
    plot_equity_curves(all_results, f"simulate_market_equity_{BASE_KWARGS.get("ts_code")}.png")

    print("\n✅ 完成！")
