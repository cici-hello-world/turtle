"""海龟策略回测配置"""

from dataclasses import dataclass, field
import pandas as pd


@dataclass
class BacktestConfig:
    """回测配置数据类，覆盖所有参数"""

    # ========== 股票信息 ==========
    ts_code: str = ""
    stock_name: str = ""
    market: str = "A"  # "A" | "HK"

    # ========== 数据参数 ==========
    years: int = 8
    start_date: str = ""  # 留空自动计算
    end_date: str = ""  # 留空自动计算
    exchange: str = ""  # 留空按 market 自动推导
    per_shares: int = 100  # 每手股数

    # ========== 策略参数 ==========
    entry_period: int = 20
    exit_period: int = 10
    atr_period: int = 14
    n_value: float = 2.0  # 止损 ATR 倍数
    risk_percent: float = 1.0  # 单笔风险 %
    initial_capital: float = 200000.0

    # ========== EMA 趋势过滤器 ==========
    ema_fast: int = 40   # 快线周期，0 表示禁用
    ema_slow: int = 200  # 慢线周期，0 表示禁用
    # 做多仅当 EMA_Fast > EMA_Slow（上升趋势），做空反之

    # ========== 通道类型 ==========
    channel_type: str = "donchian"  # "donchian" | "bollinger"
    bb_period: int = 20             # 布林线周期（仅 bollinger）
    bb_k: float = 2.0               # 布林线标准差倍数（仅 bollinger）

    # ========== 策略模式 ==========
    mode: str = "long_more"  # "long" | "long_more" | "both"
    max_pyramid: int = 5  # 最大加仓次数（仅 long_more）
    skip_first_add: bool = False  # 跳过首次加仓，总加仓次数-1（仅 long_more）

    # ========== 输出 ==========
    output_dir: str = ""  # 留空自动推导

    def __post_init__(self):
        today = pd.Timestamp.today()

        # 自动计算日期
        if not self.start_date:
            self.start_date = (today - pd.DateOffset(years=self.years)).strftime("%Y%m%d")
        if not self.end_date:
            self.end_date = today.strftime("%Y%m%d")

        # 自动推导交易所代码
        if not self.exchange:
            if self.market == "A" or self.market == "ETF":
                # A股上交所/深交所共用 XSHG 交易日历
                self.exchange = "XSHG"
            elif self.market == "HK":
                self.exchange = "XHKG"

        # 自动推导输出目录
        if not self.output_dir:
            if self.market == "A":
                suffix_map = {
                    "long": "a_stock_long_result",
                    "long_more": "a_stock_long_more_result",
                    "short_more": "a_stock_shot_more_result",
                    "both": "a_stock_both_result",
                }
            elif self.market == "ETF":
                suffix_map = {
                    "long": "ETF_long_result",
                    "long_more": "ETF_long_more_result",
                    "short_more": "ETF_shot_more_result",
                    "both": "ETF_both_result",
                }
            else:
                suffix_map = {
                    "long": "hk_stock_long_result",
                    "long_more": "hk_stock_long_more_result",
                    "short_more": "hk_stock_shot_more_result",
                    "both": "hk_stock_both_result",
                }
            self.output_dir = suffix_map.get(self.mode, f"{self.market.lower()}_stock_backtest")

    @property
    def risk_per_trade(self) -> float:
        """单笔风险比例（小数）"""
        return self.risk_percent / 100.0

    @property
    def is_hk(self) -> bool:
        return self.market == "HK"

    @property
    def is_a(self) -> bool:
        return self.market == "A"
