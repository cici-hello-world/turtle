"""技术指标计算"""

import numpy as np
import pandas as pd


def compute_indicators(
    df: pd.DataFrame,
    entry_period: int = 20,
    exit_period: int = 10,
    atr_period: int = 14,
) -> pd.DataFrame:
    """在 DataFrame 上添加海龟策略所需指标列

    添加的列: TR, ATR, HH_Entry, LL_Entry, HL_Exit, LL_Exit
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

    # Donchian 通道
    df["HH_Entry"] = df["High"].rolling(window=entry_period).max().shift(1)
    df["LL_Entry"] = df["Low"].rolling(window=entry_period).min().shift(1)
    df["HL_Exit"] = df["High"].rolling(window=exit_period).max().shift(1)
    df["LL_Exit"] = df["Low"].rolling(window=exit_period).min().shift(1)

    return df
