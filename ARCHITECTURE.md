# 海龟策略回测框架 — 架构思维导图

```
                          run_backtest.py  (入口)
                                │
         ┌──────────────────────┼──────────────────────┐
         ▼                      ▼                      ▼
   ① data.py             ② indicators.py         ③ engine.py
   数据获取                技术指标计算              回测引擎
         │                      │                      │
         ▼                      ▼                      ▼
   ④ config.py ◄──────── 共享配置 ────────► ④ config.py
                                                │
                                   ┌────────────┼────────────┐
                                   ▼            ▼            ▼
                             ⑤ statistics.py  reporter.py  输出目录
                               统计计算        报告输出     .xlsx 文件
```

---

## 一、整体流水线 (run_backtest.py)

```
run(config)  ←── 唯一入口
  │
  ├─ 1. fetch_data()          → df (日线 OHLCV + 复权因子)
  ├─ 2. compute_indicators()  → df (追加 TR/ATR/Donchian 列)
  ├─ 3. run_backtest()        → (trades[], equity_curve[])
  ├─ 4. calc_stats()          → stats{}
  └─ 5. print_summary() + save_excel_report()
```

---

## 二、模块详解

### 1. core/config.py — 配置中心

```
BacktestConfig (@dataclass)
├── 股票信息
│   ├── ts_code: str          # e.g. "000625.SZ"
│   ├── stock_name: str       # e.g. "长安汽车"
│   └── market: str           # "A" | "HK"
│
├── 数据参数
│   ├── years: int = 8        # 回测年数
│   ├── start_date: str       # 自动计算（留空）
│   ├── end_date: str         # 自动计算（留空）
│   ├── exchange: str         # 自动推导（留空）
│   ├── per_shares: int = 100 # 每手股数
│   └── dividend_info: pd.DataFrame | None  # 港股分红
│
├── 策略参数
│   ├── entry_period: int = 20  # 入场通道周期
│   ├── exit_period: int = 10   # 出场通道周期
│   ├── atr_period: int = 14    # ATR 周期
│   ├── n_value: float = 2.0    # 止损 ATR 倍数
│   ├── risk_percent: float = 1.0  # 单笔风险%
│   └── initial_capital: float = 200000
│
├── 策略模式
│   ├── mode: str = "long"    # "long" | "long_more" | "both"
│   └── max_pyramid: int = 5  # 最大加仓次数（仅 long_more）
│
├── 输出
│   └── output_dir: str       # 自动推导
│
└── 计算属性
    ├── risk_per_trade: float # risk_percent / 100
    ├── is_hk: bool
    └── is_a: bool
```

### 2. core/data.py — 数据获取

```
fetch_data(ts_code, stock_name, market, start_date, end_date, exchange, dividend_info)
│
├─ market == "A" → _fetch_a_share()
│   ├── pro.daily()         → 日线 (open/high/low/close/vol/amount)
│   ├── pro.adj_factor()    → 复权因子
│   └── _compute_qfq()      → 前复权 OHLC
│
├─ market == "HK" → _fetch_hk_share()
│   ├── pro.hk_daily_adj()  → 港股复权日线
│   ├── modify_adj_factor() → 手动修正分红除权
│   └── _compute_qfq()      → 前复权 OHLC
│
└── _check_data_completeness()  → 交易日完整性校验

输出列: Date, Open, High, Low, Close, Volume, Amount, Adj_Factor
```

### 3. core/indicators.py — 技术指标

```
compute_indicators(df, entry_period, exit_period, atr_period)
│
├── TR  (True Range)
│   └── max(H-L, |H-C_prev|, |L-C_prev|)
│
├── ATR = SMA(TR, atr_period)   # 默认14日简单移动平均
│
├── HH_Entry = max(High, entry_period).shift(1)   # 入场突破上轨 (20日)
├── LL_Entry = min(Low,  entry_period).shift(1)   # 入场突破下轨 (20日)
├── HL_Exit  = max(High, exit_period ).shift(1)   # 出场突破上轨 (10日)
└── LL_Exit  = min(Low,  exit_period ).shift(1)   # 出场突破下轨 (10日)
```

### 4. core/engine.py — 回测引擎（核心）

```
run_backtest(df, config)  ←── 统一入口
│
├── mode == "long"       → _run_long_only()
├── mode == "long_more"  → _run_long_with_pyramid()
└── mode == "both"       → _run_long_short()
```

#### 4.0 共享工具函数

```
_calc_shares(capital, risk_per_trade, stop_dist, per_shares) → int
  │  股数 = capital × risk_pct / (stop_dist × per_shares)
  └── 用途: 根据风险资金计算入场手数

_can_afford(cost, capital, ratio=0.8) → bool
  └── 用途: 检查资金是否足够（默认最多用80%资金）

_prep_entry(row, capital, config, per_shares) → (stop_dist, shares)
  │  校验 ATR → 计算 stop_dist = N × ATR → 计算 shares
  └── 返回 (0, 0) 表示无效，调用方应 continue

_build_trade(entry_date, exit_date, pos_type, entry_price, exit_price,
             shares, cost, pnl, capital, reason, *, stop_dist=0.0, more_info=None) → dict
  └── 用途: 构建统一格式的交易记录字典
      字段: Entry Date, Exit Date, Type, Entry Price, Exit Price,
            Shares, Cost_¥, PnL_¥, PnL_%, Capital_After, Reason,
            [2ATR], [More Info]

_check_long_exit(row, stop_loss_price) → (exit_price | None, reason)
  │  1. row.Close < stop_loss_price  → Hard Stop Loss (2×ATR)
  │  2. row.Close < row.LL_Exit      → Trend Reversal (10d Low)
  └── 用于: _run_long_only, _run_long_with_pyramid

_check_both_exit(row, pos, stop_loss_price) → (exit_price | None, reason)
  │  long:  Close < stop_loss  |  Close < LL_Exit
  │  short: Close > stop_loss  |  Close > HL_Exit
  └── 用于: _run_long_short
```

#### 4.1 不加仓做多 `_run_long_only()`

```
循环每天:
  ├── 无持仓:
  │   ├── _prep_entry() → stop_dist, shares
  │   └── Close > HH_Entry → 入场做多
  │       入场价=Close, 止损价=入场价-stop_dist
  │
  └── 持仓中:
      └── _check_long_exit() → 触发则平仓
          盈亏 = (出场价 - 入场价) × 股数 × 每手股数

强制平仓: 按最后收盘价平仓
```

#### 4.2 加仓做多 `_run_long_with_pyramid()`

```
循环每天:
  ├── 无持仓:
  │   ├── _prep_entry() → stop_dist, shares
  │   ├── base_shares = shares, max_shares = shares × max_pyramid
  │   └── Close > HH_Entry → 入场做多
  │
  └── 持仓中:
      ├── _check_long_exit() → 触发则平仓
      ├── 加仓条件 (仅当未触发 exit):
      │   ├── Close > 入场价 + 持有单位数 × 0.5 × ATR
      │   └── 且 当前股数 < max_shares
      │        └── 加仓: 止损上移至 加仓价 - N×ATR
      └── 平仓时:
          总成本 = 入场成本 + 所有加仓成本之和
          盈亏 = 卖出收入 - 总成本
          more_info[] 记录每次加仓的价格和日期

强制平仓: 按最后收盘价平仓, 使用 total_cost 计算盈亏
```

#### 4.3 多空双向 `_run_long_short()`

```
循环每天:
  ├── 无持仓:
  │   ├── _prep_entry() → stop_dist, shares
  │   ├── Close > HH_Entry → 做多入场 (cost > capital×0.95 则跳过)
  │   └── Close < LL_Entry → 做空入场 (收入加入 capital)
  │        short: current_shares = -shares (负数标记)
  │
  └── 持仓中:
      └── _check_both_exit() → 触发则平仓
          long:  盈亏 = (出场价-入场价) × shares × 100
          short: 盈亏 = (出场价-入场价) × (-shares) × 100
          capital += exit_price × shares × 100  (shares 已含正负号)

强制平仓: 按最后收盘价平仓
  long:  cost_basis = entry_price × shares × 100
  short: cost_basis = |last_price × shares × 100|
  capital += sell_proceeds  (sell_proceeds 已含正负号)

⚠️ 硬编码 PS=100，不使用 config.per_shares
```

#### 4.4 输出格式化函数

```
_print_header(label, initial_capital, risk_percent)   # 回测开始横幅
_print_entry(label, price, shares, cost, capital, stop_loss)       # 做多入场
_print_short_entry(label, price, shares, income, capital, stop_loss) # 做空入场
_print_pyramid_add(label, price, shares, cost, capital, stop_loss)   # 加仓
_print_exit(label, pnl, capital, reason)               # 平仓
_print_force_close(label, pnl, capital)                # 强制平仓
```

### 5. core/statistics.py — 统计计算

```
calc_stats(trades, initial_capital, equity_curve) → dict
│
├── 过滤: 排除 Reason=="强制平仓" 的交易
│
├── 胜率: win_count / total_trades × 100
├── 赔率: avg_win_¥ / avg_loss_¥
├── 总盈亏: final_capital - initial_capital
├── 总收益率: total_pnl / initial_capital × 100
├── 最大回撤: max( (equity - cummax) / cummax ) × 100
├── 夏普比率: mean(daily_return) / std(daily_return) × √252
└── 期望收益率: win_rate × 赔率 - loss_rate
```

### 6. core/reporter.py — 报告输出

```
print_summary(stats, config)
  └── 控制台打印: 交易次数/胜率/赔率/总盈亏/收益率/回撤/夏普

save_excel_report(df, trades, stats, config)
  ├── Sheet 1: Summary        — 策略汇总统计
  ├── Sheet 2: 日线原始数据    — 全部日线 OHLCV + 指标
  └── Sheet 3: System 1 交易明细 — 每笔交易的入场/出场/盈亏

输出路径: {output_dir}/Backtest_Report_{ts_code}_{start_date}起_{risk}%单笔风险.xlsx
```

---

## 三、数据流图

```
Config
  │
  ├── ts_code, start_date, end_date, market ──► data.fetch_data()
  │                                                │
  │                                          df (OHLCV + Adj_Factor)
  │                                                │
  ├── entry_period, exit_period, atr_period ──► indicators.compute_indicators()
  │                                                │
  │                                          df (追加 TR/ATR/HH_Entry/LL_Entry/HL_Exit/LL_Exit)
  │                                                │
  ├── mode, n_value, risk_per_trade,            engine.run_backtest()
  │   per_shares, max_pyramid,                        │
  │   entry_period                              trades[] + equity_curve[]
  │                                                │
  └── initial_capital ─────────────────────► statistics.calc_stats()
                                                   │
                                             stats{}
                                                   │
                                            reporter.print_summary()
                                            reporter.save_excel_report()
```

---

## 四、常见需求 → 改动位置

| 需求 | 改动文件 | 具体位置 |
|---|---|---|
| 调整入场/出场周期 | `config.py` | `entry_period` / `exit_period` |
| 调整止损倍数 | `config.py` | `n_value` |
| 调整单笔风险 | `config.py` | `risk_percent` |
| 调整每手股数 | `config.py` | `per_shares` |
| 新增策略模式（如 only_short） | `engine.py` | 新增 `_run_short_only()` + `run_backtest()` 添加分支 |
| 修改做多入场条件 | `engine.py` | 各 `_run_*` 函数中 `if row["Close"] > row["HH_Entry"]` |
| 修改做空入场条件 | `engine.py` | `_run_long_short()` 中 `elif row["Close"] < row["LL_Entry"]` |
| 修改止损逻辑 | `engine.py` | `_check_long_exit()` / `_check_both_exit()` |
| 修改加仓规则 | `engine.py` | `_run_long_with_pyramid()` 中 "加仓条件" 段 |
| 新增技术指标 | `indicators.py` | `compute_indicators()` 末尾追加 |
| 修改资金管理公式 | `engine.py` | `_calc_shares()` 或 `_can_afford()` |
| 新增统计指标 | `statistics.py` | `calc_stats()` 返回字典追加 |
| 修改报告格式 | `reporter.py` | `print_summary()` / `save_excel_report()` |
| 新增数据源（非 tushare） | `data.py` | 新增 `_fetch_*` 函数 + `fetch_data()` 添加分支 |
| 修改交易记录字段 | `engine.py` | `_build_trade()` |
| 修改输出日志格式 | `engine.py` | `_print_*` 系列函数 |
| 调整资金使用上限比例 | `engine.py` | `_can_afford(ratio=...)` 调用处 (默认0.8) |
| 多空模式使用 config.per_shares | `engine.py` | `_run_long_short()` 中 `PS = 100` → `config.per_shares` |

---

## 五、engine.py 函数调用关系图

```
run_backtest()
  ├── _run_long_only()
  │     ├── _prep_entry()
  │     │     ├── _calc_shares()
  │     │     └── (ATR 校验)
  │     ├── _can_afford()
  │     ├── _check_long_exit()
  │     ├── _build_trade()
  │     ├── _print_header()
  │     ├── _print_entry()
  │     ├── _print_exit()
  │     └── _print_force_close()
  │
  ├── _run_long_with_pyramid()
  │     ├── _prep_entry()
  │     │     ├── _calc_shares()
  │     │     └── (ATR 校验)
  │     ├── _can_afford()
  │     ├── _check_long_exit()
  │     ├── _build_trade()
  │     ├── _print_header()
  │     ├── _print_entry()
  │     ├── _print_pyramid_add()
  │     ├── _print_exit()
  │     └── _print_force_close()
  │
  └── _run_long_short()
        ├── _prep_entry()
        │     ├── _calc_shares()
        │     └── (ATR 校验)
        ├── _check_both_exit()
        ├── _build_trade()
        ├── _print_header()
        ├── _print_entry()
        ├── _print_short_entry()
        ├── _print_exit()
        └── _print_force_close()
```

---

## 六、交易记录 (trade dict) 字段说明

| 字段 | 类型 | 说明 |
|---|---|---|
| `Entry Date` | datetime | 入场日期 |
| `Exit Date` | datetime | 出场日期 |
| `Type` | str | "Long" / "Short" |
| `Entry Price` | float | 入场价 |
| `Exit Price` | float | 出场价 |
| `Shares` | int | 持仓股数（手数，short为负） |
| `Cost_¥` | float | 总成本（¥） |
| `PnL_¥` | float | 盈亏金额（¥） |
| `PnL_%` | float | 盈亏百分比 |
| `Capital_After` | float | 平仓后剩余资金 |
| `Reason` | str | 出场原因 |
| `2ATR` | float | 入场时的 2×ATR 止损距离（仅 long/long_more） |
| `More Info` | list[dict] | 加仓明细（仅 long_more） |
