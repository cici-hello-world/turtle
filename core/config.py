"""海龟策略回测配置"""

import os
from dataclasses import dataclass, field
import pandas as pd

# 项目根目录 & 输出根目录
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RESULT_ROOT = os.path.join(_PROJECT_ROOT, "result")


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
    data_source: str = "tushare"  # "tushare" | "akshare" | "sina" | "baostock"

    # ========== 策略参数 ==========
    entry_period: int = 20
    exit_period: int = 10
    atr_period: int = 14
    n_value: float = 2.0  # 止损 ATR 倍数
    risk_percent: float = 1.0  # 单笔风险 %
    initial_capital: float = 200000.0

    # ========== DMI 趋势过滤器 ==========
    user_dmi: bool = False  # 默认不启用

    # ========== EMA 趋势过滤器 ==========
    ema_fast: int = 40   # 快线周期，0 表示禁用
    ema_slow: int = 200  # 慢线周期，0 表示禁用
    # 做多仅当 EMA_Fast > EMA_Slow（上升趋势），做空反之

    # ========== 通道类型 ==========
    channel_type: str = "donchian"  # "donchian" | "bollinger"
    bb_period: int = 20             # 布林线周期（仅 bollinger）
    bb_k: float = 2.0               # 布林线标准差倍数（仅 bollinger）

    # ========== 策略模式 ==========
    mode: str = "long_more"  # "long" | "long_more" | "short_more" | "both" | "anti_martingale" | "position_management_1" | "position_management_2"
    max_pyramid: int = 4  # 最大加仓次数（含初始仓位，Turtle 原版单一市场上限 4 单位）
    skip_first_add: bool = False  # 跳过首次加仓，总加仓次数-1（仅 long_more）

    # ========== Anti-Martingale 参数（仅 anti_martingale 模式） ==========
    am_ma_period: int = 50               # MA 均线周期，用于入场/出场信号
    am_risk_multiplier: float = 1.5      # 盈利后风险乘数
    am_max_multiplier: float = 4.0       # 最大风险乘数上限
    # ========== donchian 参数（网格仓位管理） ==========
    don_start_date: str = ""          # donchian起始日期，格式 "YYYYMMDD"

    # ========== Position Management 1 参数（网格仓位管理） ==========
    pm_start_date: str = ""          # 仓位管理起始日期，格式 "YYYYMMDD"
    pm_initial_shares: int = 0       # 初始持仓手数
    pm_reduce_pct: float = 0.25      # 卖出权益的百分比
    pm_add_pct: float = 0.15         # 买入权益的百分比
    pm_grid_up: float = 0.05         # 相对参考价上涨X%触发卖出
    pm_grid_down: float = 0.05       # 相对参考价下跌X%触发买入
    pm_max_position_pct: float = 0.9  # 仓位上限（持仓市值/权益）
    pm_min_cash_pct: float = 0.10      # 现金下限（现金/权益）
    pm_use_dmi: bool = False          # 是否启用 DMI 过滤器
    pm_adaptive: bool = True         # 趋势自适应：单边上涨趋势转入donchian通道
    pm_dynamic_position: bool = True  # 动态仓位上限：根据波动率(ATR/价格)自动调整 max_position_pct
    pm_dyn_vol_1: float = 0.03        # 波动率分档1：<此值为低波动
    pm_dyn_vol_2: float = 0.06        # 波动率分档2：<此值为中波动
    pm_dyn_vol_3: float = 0.10        # 波动率分档3：<此值为高波动（超过为极端波动）
    pm_dyn_pos_1: float = 0.90        # 低波动仓位上限
    pm_dyn_pos_2: float = 0.80        # 中波动仓位上限
    pm_dyn_pos_3: float = 0.60        # 高波动仓位上限
    pm_dyn_pos_4: float = 0.40        # 极端波动仓位上限

    # ========== 自适应网格参数（ADX < 25 时切换到激进参数） ==========
    pm_adaptive_grid: bool = True    # 是否启用自适应网格
    pm_adaptive_interp: bool = True  # 不对称插值（False=跳过插值公式，RSI/BB乘数直接作用于静态网格）
    pm_adx_threshold: float = 25.0   # ADX 震荡/趋势分界线
    pm_agg_k: float = 0.6            # 震荡市激进倍数: gda=gd×k, gua=gu×k, rpa=rp+(1-k)×0.5

    # ========== RSI 超买超卖过滤器（叠加于 DMI 之上） ==========
    pm_rsi_enabled: bool = True      # 是否启用 RSI 过滤器
    pm_rsi_period: int = 14          # RSI 计算周期
    pm_rsi_oversold: float = 30.0    # 超卖阈值（RSI低于此值 → 更积极买入、禁止卖出）
    pm_rsi_overbought: float = 70.0  # 超买阈值（RSI高于此值 → 禁止买入、更积极卖出）

    # ========== BB 宽度自适应网格（叠加于 DMI+RSI 之上，仅缩放网格间距） ==========
    pm_bb_enabled: bool = False      # 是否启用 BB 宽度联动网格（宽带→网格放宽，窄带→网格收紧）
    pm_bb_period: int = 20           # BB 计算周期
    pm_bb_k: float = 2.0             # BB 标准差倍数
    pm_bb_width_lookback: int = 252  # BB 宽度 Z-score 归一化滚动窗口
    pm_bb_width_min: float = 0.6     # 窄带(percentile≈0)时的网格乘数（收紧网格）
    pm_bb_width_max: float = 1.5     # 宽带(percentile≈1)时的网格乘数（放宽网格）

    # ========== 止盈与回本参数 ==========
    pm_take_profit_ratio: float = 100  # 权益超过初始×该倍数后：仓位归零时停止交易，有仓位时全仓清仓

    # ========== Position Management 2 参数（Swing 恢复仓位管理） ==========
    # 与 PM1 不同：PM2 做非对称 swing —— 反弹卖出、回落买回，靠波动差价摊薄成本
    # 卖出和买回的振幅不对等，交易量是固定手数而非权益%
    pm2_bounce_up: float = 0.10      # 从近期低点反弹X%触发卖出
    pm2_buyback_down: float = 0.08   # 从卖出价回落X%触发买回（必须< bounce_up 才有利润）
    pm2_trade_lots: int = 1          # 每次交易固定手数

    # ========== 输出 ==========
    output_dir: str = ""  # 留空自动推导

    def __post_init__(self):
        today = pd.Timestamp.today()

        # —— 自适应网格约束：k 在合理范围内 ——
        if self.pm_adaptive_grid and self.mode == "position_management_1":
            if self.pm_agg_k <= 0 or self.pm_agg_k > 1.0:
                print(f"⚠️ [配置修正] pm_agg_k({self.pm_agg_k}) 超出 (0,1]，已重置为 0.7")
                self.pm_agg_k = 0.7

        # 自动计算日期
        if not self.start_date:
            self.start_date = (today - pd.DateOffset(years=self.years)).strftime("%Y%m%d")
        if not self.end_date:
            self.end_date = today.strftime("%Y%m%d")
            if pd.Timestamp.now().hour < 18:
                # 晚上6点之后才拉取当前的，否则值拉取到前一天
                self.end_date = (today - pd.DateOffset(days=1)).strftime("%Y%m%d")

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
                    "anti_martingale": "a_stock_anti_martingale_result",
                    "position_management_1": "position_management_result",
                }
            elif self.market == "ETF":
                suffix_map = {
                    "long": "ETF_long_result",
                    "long_more": "ETF_long_more_result",
                    "short_more": "ETF_shot_more_result",
                    "both": "ETF_both_result",
                    "anti_martingale": "ETF_anti_martingale_result",
                    "position_management_1": "position_management_result",
                }
            else:
                suffix_map = {
                    "long": "hk_stock_long_result",
                    "long_more": "hk_stock_long_more_result",
                    "short_more": "hk_stock_shot_more_result",
                    "both": "hk_stock_both_result",
                    "anti_martingale": "hk_stock_anti_martingale_result",
                    "position_management_1": "position_management_result",
                }
            sub_dir = suffix_map.get(self.mode, f"{self.market.lower()}_stock_backtest")
            self.output_dir = os.path.join(_RESULT_ROOT, sub_dir)

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
