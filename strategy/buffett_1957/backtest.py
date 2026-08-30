#!/usr/bin/env python3
"""组合回测循环：月度估值权重 + 季度价值换仓 + 现金等待（套利类 MVP）

主日历 = 恒指交易日；成员价格 reindex+ffill；除权日分红加回现金。
交易成本：core.engine._pm_trade_cost（港股 0.12% 双向）。
交易记录为 PM 式（含 Position_After=当期价值权重），供 calc_stats 统计。
"""
import numpy as np
import pandas as pd

from strategy.buffett_1957.config import Buffett1957Config
from strategy.buffett_1957.hsi_pe import load_hsi_price, get_monthly_weights
from strategy.buffett_1957.hk_valuation import build_quarterly_selection
from strategy.buffett_1957.hk_prices import collect_prices
from strategy.buffett_1957.hk_universe import load_hk_basic
from core.engine import _pm_trade_cost


def run_backtest(config: Buffett1957Config, weights: pd.Series | None = None,
                 value_only: bool = False,
                 selections: dict | None = None,
                 prices: dict | None = None) -> tuple[list[dict], list[float], list]:
    """执行组合回测。value_only=True 时权重恒为 1（100% 价值组合对照）。

    返回 (trades, equity_curve, equity_dates)，与 core.engine.run_backtest 同构。
    """
    hsi = load_hsi_price()
    hsi = hsi[(hsi["Date"] >= pd.to_datetime(config.start_date))
              & (hsi["Date"] <= pd.to_datetime(config.end_date))].reset_index(drop=True)
    calendar = hsi["Date"].tolist()

    if weights is None:
        weights, _, _ = get_monthly_weights(config)
    if selections is None:
        selections = build_quarterly_selection(config)
    member_codes = sorted({c for s in selections.values() for c in s["ts_code"]})
    if prices is None:
        prices = collect_prices(member_codes, config.start_date, config.end_date)

    basic = load_hk_basic()
    unit_map = dict(zip(basic["ts_code"], basic["trade_unit"]))

    # 成员价格面板：reindex 到主日历，ffill（停牌净值走平）
    close_cols, div_cols = {}, {}
    for code, df in prices.items():
        s = df.set_index("Date")["Close"]
        s = s[~s.index.duplicated(keep="last")].reindex(calendar).ffill()
        close_cols[code] = s
        d = df.set_index("Date")["Cash_Dividend"]
        d = d[~d.index.duplicated(keep="last")].reindex(calendar).fillna(0.0)
        div_cols[code] = d
    close_panel = pd.DataFrame(close_cols, index=calendar)
    div_panel = pd.DataFrame(div_cols, index=calendar)

    # 月度权重对齐到主日历（月末最后交易日执行再平衡）
    w_daily = pd.Series(np.nan, index=calendar)
    if value_only:
        w_daily[:] = 1.0
    else:
        weights = weights.sort_index()
        month_ends = pd.Series(calendar, index=calendar).resample("ME").last()
        for t, d in month_ends.items():
            t = pd.Timestamp(t)
            idx = weights.index[weights.index <= t]
            if len(idx) > 0:
                w_daily.loc[d] = weights.loc[idx[-1]]
        w_daily = w_daily.ffill().fillna(1.0)

    # 季度换仓日历
    quarter_dates = set(selections.keys())

    cash = config.initial_capital
    holdings: dict[str, int] = {}          # ts_code → 股数
    trades: list[dict] = []
    equity_curve: list[float] = []
    equity_dates: list = []

    def _market_value(d: pd.Timestamp) -> float:
        v = 0.0
        for code, sh in holdings.items():
            px = close_panel.at[d, code]
            v += sh * px if not pd.isna(px) else 0.0
        return v

    def _trade_record(d, ttype, code, shares, price, amount, pos_after, equity_after, reason):
        trades.append({
            "Date": d, "Entry Date": d, "Exit Date": d,
            "Type": ttype, "Price": price, "Shares": shares,
            "Entry Price": price, "Exit Price": price,
            "Cost_¥": abs(amount) * 0.0012, "PnL_¥": amount if ttype == "减仓" else -amount,
            "PnL_%": 0.0, "Capital_After": cash,
            "Reason": reason, "Amount": amount,
            "Position_After": pos_after, "Equity_After": equity_after,
            "Reference_Price": price, "more_info": [],
        })

    for d in calendar:
        # 除权日分红簿记（不复权价格当日自然下跌）
        for code, sh in list(holdings.items()):
            dv = div_panel.at[d, code] if code in div_panel.columns else 0.0
            if dv and dv > 0 and sh > 0:
                cash += dv * sh

        w = float(w_daily.loc[d])
        is_quarter_end = d in quarter_dates
        is_month_end = d == pd.Series(calendar, index=calendar).resample("ME").last().get(d, None)

        if (is_quarter_end or is_month_end):
            nav = cash + _market_value(d)
            target_value = w * nav

            if is_quarter_end and d in selections:
                # 季度换仓：卖出全部旧成员
                for code, sh in list(holdings.items()):
                    px = close_panel.at[d, code]
                    if pd.isna(px):
                        px = 0.0
                    gross = sh * px
                    cost = _pm_trade_cost(gross, "HK", "sell")
                    cash += gross - cost
                    _trade_record(d, "减仓", code, sh, px, gross - cost,
                                  w, cash, f"季度换仓卖出 {code}")
                holdings = {}
                # 等权买入新成员
                sel = selections[d]
                n = max(len(sel), 1)
                budget_per = target_value / n
                for _, row in sel.iterrows():
                    code = row["ts_code"]
                    px = close_panel.at[d, code] if code in close_panel.columns else np.nan
                    if pd.isna(px) or px <= 0:
                        continue
                    unit = unit_map.get(code, 100)
                    lots = int(budget_per / (px * unit * 1.0012))
                    if lots <= 0:
                        continue
                    gross = px * lots * unit
                    cost = _pm_trade_cost(gross, "HK", "buy")
                    if gross + cost > cash:
                        continue
                    cash -= gross + cost
                    holdings[code] = holdings.get(code, 0) + lots * unit
                    _trade_record(d, "加仓", code, lots * unit, px, -(gross + cost),
                                  w, cash + _market_value(d), f"季度换仓买入 {code}")
            else:
                # 月度再平衡：按比例缩放既有持仓至目标权重
                cur_value = _market_value(d)
                if cur_value > 0:
                    scale = target_value / cur_value
                    for code, sh in list(holdings.items()):
                        px = close_panel.at[d, code]
                        if pd.isna(px):
                            continue
                        unit = unit_map.get(code, 100)
                        target_shares = int(round(sh * scale / unit)) * unit
                        diff = target_shares - sh
                        if diff == 0:
                            continue
                        if diff > 0:      # 增持
                            gross = px * diff
                            cost = _pm_trade_cost(gross, "HK", "buy")
                            if gross + cost > cash:
                                continue
                            cash -= gross + cost
                            holdings[code] = target_shares
                            _trade_record(d, "加仓", code, diff, px, -(gross + cost),
                                          w, cash + _market_value(d), "月度加仓")
                        else:             # 减持
                            sell_sh = -diff
                            gross = px * sell_sh
                            cost = _pm_trade_cost(gross, "HK", "sell")
                            cash += gross - cost
                            holdings[code] = target_shares
                            _trade_record(d, "减仓", code, sell_sh, px, gross - cost,
                                          w, cash + _market_value(d), "月度减仓")

        equity = cash + _market_value(d)
        equity_curve.append(equity)
        equity_dates.append(d)

    return trades, equity_curve, equity_dates


if __name__ == "__main__":
    from core.statistics import calc_stats
    cfg = Buffett1957Config()
    trades, eq, eq_dates = run_backtest(cfg)
    hsi = load_hsi_price()
    hsi = hsi[(hsi["Date"] >= pd.to_datetime(cfg.start_date))
              & (hsi["Date"] <= pd.to_datetime(cfg.end_date))]
    bench = (hsi["Close"].iloc[-1] / hsi["Close"].iloc[0] - 1) * 100
    stats = calc_stats(trades, cfg.initial_capital, eq, eq_dates,
                       benchmark_return=bench)
    print(f"巴菲特组合: 收益 {stats['总收益率 (%)']:.1f}%  MaxDD {stats['最大回撤 (%)']:.1f}%  "
          f"稳健夏普 {stats['稳健夏普比率']:.2f}  交易 {stats['总交易']}  "
          f"期末 {stats['最终资金 (¥)']:,.0f}（恒指 {bench:.1f}%）")
