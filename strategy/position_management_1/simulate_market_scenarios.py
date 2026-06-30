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
#    pm_agg_k           = 0.5,    # gda=0.06 gua=0.015 rpa=0.50
# )
# 理想最优参数
# BEST_PARAMS = dict(
#    pm_grid_down       = 0.05,
#    pm_grid_up         = 0.03,
#    pm_reduce_pct      = 0.25,
#    pm_add_pct         = 0.01,
#    pm_adx_threshold   = 25,
#    pm_agg_k           = 0.7,    # gda=0.035 gua=0.021 rpa=0.40
#     pm_take_profit_ratio = 1.30,
# )
# 泡泡玛特最优参数
BEST_PARAMS = dict(
   pm_grid_down       = 0.05,
   pm_grid_up         = 0.04,
   pm_reduce_pct      = 0.5,
   pm_add_pct         = 0.3,
   pm_adx_threshold   = 25,
   pm_agg_k           = 0.6,    # gda=0.072 gua=0.042 rpa=0.45
)

# 美团最优参数
# BEST_PARAMS = dict(
#     pm_grid_down=0.07,
#     pm_grid_up = 0.05,
#     pm_reduce_pct = 0.25,
#     pm_add_pct = 0.03,
#     pm_adx_threshold = 25,
#     pm_agg_k = 0.6,        # gda=0.042 gua=0.030 rpa=0.45
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
    pm_start_date="20260629",
    ema_slow = 0,
    ema_fast = 0,
    pm_initial_shares=2,
    initial_capital=40000,
    user_dmi= True,
    pm_dynamic_position = False,
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


# ═══════════════════════════════════════════════════════════════
# 场景目标特征 (126天总变化%, 日波动%, 涨日比%)
# ═══════════════════════════════════════════════════════════════
_SCENARIO_TARGET = {
    "up":              (65.0, 1.40, 58),
    "down":            (-40.0, 1.40, 35),
    "volatile_up":     (21.0, 2.50, 50),
    "volatile_down":   (-12.0, 2.60, 42),
    "sideways":        (0.0, 1.80, 50),
    "v_reversal":      (0.0, 1.80, 48),
    "inv_v_reversal":  (0.0, 1.80, 52),
    "slow_grind_down": (-20.0, 1.50, 40),
}


def _find_best_historical_window(close: np.ndarray, target_chg: float,
                                 target_vol: float, target_up: float,
                                 window: int = 126, step: int = 10) -> tuple:
    """扫描历史，找到与目标特征最相似的 window 天区间。"""
    best_score, best = -1.0, None
    for s in range(0, len(close) - window, step):
        e = s + window
        win = close[s:e]
        rets = np.diff(win) / win[:-1]
        chg = (win[-1] / win[0] - 1) * 100
        vol = np.std(rets) * 100
        up = (rets > 0).sum() / len(rets) * 100
        score = (max(0.0, 1.0 - abs(chg - target_chg) / 50) * 0.4
                 + max(0.0, 1.0 - abs(vol - target_vol) / 2.0) * 0.3
                 + max(0.0, 1.0 - abs(up - target_up) / 25) * 0.3)
        if score > best_score:
            best_score = score
            best = (s, e, chg, vol, up)
    return (*best, best_score)


def _bootstrap_returns(close: np.ndarray, s_idx: int, e_idx: int,
                       target_slope: float, n_days: int,
                       rng: np.random.Generator,
                       v_reverse: bool = False) -> np.ndarray:
    """从历史区间提取日收益 → 打乱 → 调整均值 → 生成价格路径（从1.0起）。"""
    hist_rets = np.diff(close[s_idx:e_idx]) / close[s_idx:e_idx - 1]
    shuffled = rng.choice(hist_rets, size=n_days, replace=True)
    if v_reverse:
        # V形/倒V：后半段取反
        half = n_days // 2
        adjusted = np.concatenate([
            shuffled[:half] + target_slope - np.mean(shuffled[:half]),
            -shuffled[half:] + target_slope - np.mean(-shuffled[half:]),
        ])
    else:
        adjusted = shuffled + target_slope - np.mean(shuffled)
    return np.cumprod(1 + adjusted), adjusted


def _wrap_ohlcv(dates: pd.DatetimeIndex, close_path: np.ndarray,
                start_price: float, rng: np.random.Generator) -> pd.DataFrame:
    """给定 Close 序列，生成完整 OHLCV DataFrame。"""
    n = len(close_path)
    prev = np.concatenate([[start_price], close_path[:-1]])
    daily_ret = close_path / prev - 1
    daily_vol = np.abs(daily_ret) + 0.005
    opens = np.zeros(n)
    highs = np.zeros(n)
    lows = np.zeros(n)
    for i in range(n):
        c = close_path[i]
        gap = rng.normal(0, daily_vol[i] * 0.3)
        o = prev[i] * (1 + gap)
        spread = abs(rng.normal(0, daily_vol[i] * 0.5))
        highs[i] = max(o, c) + spread
        lows[i] = min(o, c) - spread
        if lows[i] <= 0:
            lows[i] = min(o, c) * 0.98
        opens[i] = o
    avg_vol = 50000
    volumes = rng.lognormal(np.log(avg_vol), 0.4, n).astype(int)
    amounts = (volumes * close_path * 100 / 1000).round(2)
    return pd.DataFrame({
        "ts_code": "09992.HK",
        "Date": dates,
        "Open": opens.round(2),
        "High": highs.round(2),
        "Low": lows.round(2),
        "Close": close_path.round(2),
        "Volume": volumes,
        "Amount": amounts,
        "Cash_Dividend": 0.0,
    })


def generate_ohlcv(start_date: str, days: int, start_price: float,
                   trend_type: str) -> pd.DataFrame:
    """从历史真实行情中 bootstrap 生成模拟日线 OHLCV。

    扫描指定标的全量历史，找到与场景特征最相似的 days 天区间，
    提取该区间日收益分布，打乱后调整均值，生成模拟价格路径和 OHLCV。
    """
    from core.cache import load as cache_load

    target = _SCENARIO_TARGET.get(trend_type)
    if target is None:
        raise ValueError(f"未知趋势类型: {trend_type}")
    target_chg, target_vol, target_up = target
    target_slope = (1 + target_chg / 100) ** (1 / days) - 1

    # 加载历史数据
    cached = cache_load(BASE_KWARGS.get("ts_code"), BASE_KWARGS.get("market"))
    if cached is None or cached.empty:
        raise ValueError("无法加载历史缓存数据")
    close = cached["Close"].values.astype(float)
    dates_full = pd.to_datetime(cached["Date"].values)

    rng = np.random.default_rng({
        "up": 42, "down": 43,
        "volatile_up": 44, "volatile_down": 45,
        "sideways": 46, "v_reversal": 47,
        "inv_v_reversal": 48, "slow_grind_down": 49,
    }[trend_type])

    # 找最相似的历史区间
    s_idx, e_idx, actual_chg, actual_vol, actual_up, score = \
        _find_best_historical_window(close, target_chg, target_vol, target_up, days)
    hist_start = dates_full[s_idx].strftime('%Y-%m-%d')
    hist_end = dates_full[e_idx - 1].strftime('%Y-%m-%d')

    # bootstrap 生成价格路径
    is_v = trend_type in ("v_reversal", "inv_v_reversal")
    path, daily_ret = _bootstrap_returns(close, s_idx, e_idx, target_slope, days,
                                         rng, v_reverse=is_v)
    # V形先跌后涨，倒V先涨后跌
    if trend_type == "v_reversal":
        half = days // 2
        path[:half] = path[:half] * (1 - 0.01 * np.arange(half, 0, -1) / half)[::-1]
    elif trend_type == "inv_v_reversal":
        half = days // 2
        path[half:] = path[half:] * (1 - 0.01 * np.arange(half) / half)

    close_series = start_price * path

    # 生成 OHLCV
    dates = pd.bdate_range(start=start_date, periods=days)
    df = _wrap_ohlcv(dates, close_series, start_price, rng)
    df["Date"] = pd.to_datetime(df["Date"])

    sim_chg = (close_series[-1] / close_series[0] - 1) * 100
    print(f"     📊 {trend_type}: ¥{close_series[0]:.1f}→¥{close_series[-1]:.1f} ({sim_chg:+.1f}%)  "
          f"匹配 {hist_start}→{hist_end} ({actual_chg:+.0f}%/σ{actual_vol:.1f}% 得分{score:.2f})")

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
