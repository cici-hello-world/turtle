"""技术指标计算"""

import numpy as np
import pandas as pd


def compute_indicators(
    df: pd.DataFrame,
    entry_period: int = 20,
    exit_period: int = 10,
    atr_period: int = 14,
    ema_fast: int = 40,
    ema_slow: int = 200,
    channel_type: str = "donchian",
    bb_period: int = 20,
    bb_k: float = 2.0,
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

    # ATR (简单移动平均)
    df["ATR"] = df["TR"].rolling(window=atr_period, min_periods=atr_period).mean()

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

    return df
