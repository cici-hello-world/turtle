"""技术指标计算"""

import numpy as np
import pandas as pd


def _wilder_smooth(series: np.ndarray, period: int) -> np.ndarray:
    """Wilder 平滑：前 N 期 SMA 初始化，之后递归 S_t = S_{t-1} + (X_t - S_{t-1}) / N"""
    result = np.full(len(series), np.nan)
    if len(series) <= period:
        return result
    # 找到第一个非 NaN 的位置，以此处 SMA 作为初始值
    init_start = None
    for i in range(period - 1, len(series)):
        valid = series[i - period + 1:i + 1]
        valid = valid[~np.isnan(valid)]
        if len(valid) > 0:
            result[i] = float(np.mean(valid))
            init_start = i
            break
    if init_start is None:
        return result
    for i in range(init_start + 1, len(series)):
        if np.isnan(series[i]):
            result[i] = result[i - 1]
        else:
            result[i] = result[i - 1] + (series[i] - result[i - 1]) / period
    return result


def compute_indicators(
    df: pd.DataFrame,
    entry_period: int = 20,
    exit_period: int = 10,
    atr_period: int = 20,
    ema_fast: int = 40,
    ema_slow: int = 200,
    channel_type: str = "donchian",
    bb_period: int = 20,
    bb_k: float = 2.0,
    am_ma_period: int = 0,
    mode: str = '',
) -> pd.DataFrame:
    """在 DataFrame 上添加海龟策略所需指标列

    添加的列: TR, ATR, HH_Entry, LL_Entry, HL_Exit, LL_Exit,
              EMA_Fast, EMA_Slow
    当 channel_type == "bollinger" 时额外添加: BB_Mid, BB_Upper, BB_Lower
    """
    df = df.copy()

    # True Range
    df["TR"] = np.maximum(
        df["High"] - df["Low"],
        np.maximum(
            abs(df["High"] - df["Close"].shift(1)),
            abs(df["Low"] - df["Close"].shift(1)),
        ),
    )

    # ATR (Wilder 平滑，等价于海龟 N 值: N = (19×PDN + TR) / 20)
    tr_values = df["TR"].values.astype(float)
    df["ATR"] = _wilder_smooth(tr_values, atr_period)
    # 布林线通道（仅在需要时计算）
    if channel_type == "bollinger":
        df["BB_Mid"] = df["Close"].rolling(window=bb_period, min_periods=bb_period).mean()
        bb_std = df["Close"].rolling(window=bb_period, min_periods=bb_period).std()
        df["BB_Upper"] = df["BB_Mid"] + bb_k * bb_std
        df["BB_Lower"] = df["BB_Mid"] - bb_k * bb_std
    elif channel_type == "donchian":
        # Donchian 通道
        df["HH_Entry"] = df["High"].rolling(window=entry_period).max().shift(1)
        df["LL_Entry"] = df["Low"].rolling(window=entry_period).min().shift(1)
        df["HL_Exit"] = df["High"].rolling(window=exit_period).max().shift(1)
        df["LL_Exit"] = df["Low"].rolling(window=exit_period).min().shift(1)
        # EMA 趋势过滤器（0 表示禁用）
        if ema_fast > 0:
            df["EMA_Fast"] = df["Close"].ewm(span=ema_fast, adjust=False).mean()
        if ema_slow > 0:
            df["EMA_Slow"] = df["Close"].ewm(span=ema_slow, adjust=False).mean()

    # Anti-Martingale MA（仅在 am_ma_period > 0 时计算，与通道类型无关）
    if mode == "anti_martingale" and am_ma_period > 0:
        df["AM_MA"] = df["Close"].rolling(window=am_ma_period, min_periods=am_ma_period).mean()
    # DMI 指标（同花顺/通达信 SMA 平滑方法，与 Choice 对齐）
    if mode == "position_management_1" or channel_type == "donchian":
        di_period = 14
        adx_period = 6

        high = df["High"].values.astype(float)
        low = df["Low"].values.astype(float)
        close = df["Close"].values.astype(float)

        # ---- 1. +DM / -DM raw ----
        prev_high = np.roll(high, 1)
        prev_high[0] = high[0]
        prev_low = np.roll(low, 1)
        prev_low[0] = low[0]
        h_diff = high - prev_high
        l_diff = prev_low - low
        plus_dm = np.where((h_diff > l_diff) & (h_diff > 0), h_diff, 0.0)
        minus_dm = np.where((l_diff > h_diff) & (l_diff > 0), l_diff, 0.0)

        # ---- 2. True Range（修复 TR[0]=NaN） ----
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        tr_raw = np.maximum(
            high - low,
            np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)),
        )

        # ---- 3. SMA 平滑（同花顺/通达信方法，min_periods=1 允许首日开始输出） ----
        def _sma(series, period):
            return pd.Series(series).rolling(period, min_periods=1).mean().values

        tr_ma = _sma(tr_raw, di_period)
        df["+DI"] = np.where(tr_ma > 0, 100 * _sma(plus_dm, di_period) / tr_ma, 0.0)
        df["-DI"] = np.where(tr_ma > 0, 100 * _sma(minus_dm, di_period) / tr_ma, 0.0)

        # ---- 4. ADX = SMA(DX, adx_period) ----
        di_sum = df["+DI"] + df["-DI"]
        dx = np.where(di_sum > 0, 100 * np.abs(df["+DI"].values - df["-DI"].values) / di_sum, np.nan)
        df["ADX14"] = _sma(dx, adx_period)

        # ---- 5. ADXR = (ADX + ADX.shift(N)) / 2 ----
        df["ADXR"] = (df["ADX14"] + df["ADX14"].shift(adx_period)) / 2
        df["ADX_UP"] = df["ADX14"] > df["ADX14"].shift(1)
        df["ADX_CROSS_UP_ADXR"] = (df["ADX14"] > df["ADXR"]) & (df["ADX14"].shift(1) <= df["ADXR"].shift(1))
        df["ADX_DOWN"] = df["ADX14"] < df["ADX14"].shift(1)
        df["ADX_CROSS_DOWN_ADXR"] = (df["ADX14"] < df["ADXR"]) & (df["ADX14"].shift(1) >= df["ADXR"].shift(1))

        # ADX 5日斜率（连续变量，替代二元 ADX_UP 用于趋势强度评分）
        adx14 = df["ADX14"].values.astype(float)
        adx_5d_ago = np.roll(adx14, 5)
        adx_5d_ago[np.isnan(adx_5d_ago)] = adx14[np.isnan(adx_5d_ago)]
        denom = np.where(np.abs(adx_5d_ago) > 0.1, np.abs(adx_5d_ago), 1.0)
        df["ADX_SLOPE_5"] = (adx14 - adx_5d_ago) / denom

    # ── 个股 DMI 基线：自动计算，供 _dmi_trend_area 自适应分区 ──
    df.attrs["dmi_baseline"] = _compute_dmi_baseline(df)

    return df


def _compute_dmi_baseline(df: pd.DataFrame) -> dict:
    """从个股历史 DMI 数据自动计算分区阈值基线。

    ADX_SLOPE_5 是归一化指标（ΔADX/|ADX_5d_ago|），分布跨个股一致，
    slope_sep 使用硬编码 -0.05 即可。DI 价差分布有个股差异，di_gap 和 di_dir
    用该个股自身的百分位数。

    返回的 dict 存入 df.attrs["dmi_baseline"]，引擎 _dmi_trend_area()
    自动读取，无需额外配置。
    """
    dmi_cols = ["ADX_SLOPE_5", "+DI", "-DI"]
    available = [c for c in dmi_cols if c in df.columns]
    if len(available) < 3:
        return {}  # DMI 未计算，返回空

    slope = df["ADX_SLOPE_5"].dropna()
    plus_di = df["+DI"].dropna()
    minus_di = df["-DI"].dropna()
    di_abs = abs(plus_di - minus_di).dropna()

    if len(slope) < 20 or len(di_abs) < 20:
        return {}

    return {
        # ADX 斜率分界：归一化指标，-0.05 跨个股通用
        "slope_sep": -0.05,
        # DI 强势阈值：个股自身 P40，DI 价差大的个股自动调高门槛
        "di_gap": round(float(di_abs.quantile(0.40)), 1),
        # DI 方向阈值：个股自身 P20，敏感度与波动特征匹配
        "di_dir": round(float(di_abs.quantile(0.20)), 1),
    }
