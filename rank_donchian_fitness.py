# #!/usr/bin/env python3
# """TOP10 唐奇安策略适配度排名 —— EMA(40,200) 加仓做多
#
# 数据源:
#   港股: Tushare hk_daily_adj (adj_factor) + akshare (分红补权)
#   A股:  Tushare daily + adj_factor → 前复权
# 回测区间: 每只股票从上市首日 → 今日
#
# 输出:
#   top10_fitness_ranking_{a/hk}.csv
#   {a/hk}_daily_data/{code}.xlsx
#
# 用法:
#   python rank_donchian_fitness.py           # 默认港股
#   python rank_donchian_fitness.py A         # A股
# """
#
# import os, re, sys, pandas as pd, numpy as np, warnings
# import tushare as ts
#
# warnings.filterwarnings("ignore")
#
# # ============================================================
# # 配置
# # ============================================================
# MARKET = sys.argv[1].upper() if len(sys.argv) > 1 else "HK"
# assert MARKET in ("HK", "A"), f"不支持的市场: {MARKET}"
# # 设置环境变量让 optimize_donchian_ema 感知市场
# os.environ["TURTLE_MARKET"] = MARKET
#
# END_DATE = "20260608"
# INITIAL_CAPITAL = 200000.0
# _suffix = "a" if MARKET == "A" else "hk"
# OUTPUT = os.path.join(os.path.dirname(__file__), f"top10_fitness_ranking_{_suffix}.csv")
#
# # 唐奇安 + EMA 参数
# ENTRY_PERIOD = 20
# EXIT_PERIOD = 10
# ATR_PERIOD = 14
# EMA_FAST = 40
# EMA_SLOW = 200
# N_VALUE = 2.0
# RISK_PERCENT = 0.01
# PER_SHARES = 100
# MAX_PYRAMID = 5
# SKIP_FIRST_ADD = False
#
# # 市场相关 import
# if MARKET == "HK":
#     import akshare as ak
#     from optimize_donchian_ema import HK_CANDIDATES, START_DATE as LIQ_START
#     CANDIDATES = HK_CANDIDATES
# else:
#     from optimize_donchian_ema import A_CANDIDATES, START_DATE as LIQ_START
#     CANDIDATES = A_CANDIDATES
#
# # fetch_top10_by_liquidity 根据 TURTLE_MARKET 环境变量分发
# from optimize_donchian_ema import fetch_top10_by_liquidity,run_parameter_scan
#
#
#
# # ============================================================
# # 港股: 分红解析 & adj_factor 补权
# # ============================================================
# def parse_dividend_hkd(scheme: str) -> float | None:
#     """从 akshare 分红方案文本中提取港币每股现金分红"""
#     if not isinstance(scheme, str):
#         return None
#     # "相当于港币X.XX元" 或 "相当于港币X.XX元(计算值)"
#     m = re.search(r'相当于港币([\d.]+)元', scheme)
#     if m:
#         return float(m.group(1))
#     # "每股派港币X.XX元"
#     m = re.search(r'每股派港币([\d.]+)元', scheme)
#     if m:
#         return float(m.group(1))
#     # "每股派人民币...元(相当于港币X.XX元)" —— 已在上面匹配
#     return None
#
#
# def find_missing_dividends(pro, code: str, df_adj: pd.DataFrame) -> list[tuple[str, float]]:
#     """对比 Tushare adj_factor 最后变动日期与 akshare 分红记录，
#     返回缺失的 (ex_date_yyyymmdd, dividend_hkd) 列表。"""
#     # Tushare adj_factor 最后变动日期
#     d = df_adj.copy()
#     d['adj_diff'] = d['adj_factor'].diff().abs()
#     changes = d[d['adj_diff'] > 1e-8]
#     last_adj_date = str(changes['trade_date'].iloc[-1])[:10].replace('-', '') \
#         if not changes.empty else None
#
#     # akshare 分红
#     ak_code = code.replace('.HK', '')
#     try:
#         div = ak.stock_hk_dividend_payout_em(symbol=ak_code)
#     except Exception:
#         return []
#
#     missing = []
#     today_str = END_DATE
#     for _, r in div.iterrows():
#         ex_date_str = str(r['除净日']).replace('-', '')[:8]
#         scheme = str(r['分红方案'])
#
#         # 跳过未来日期
#         if ex_date_str > today_str:
#             continue
#         # 跳过实物分派（非现金）
#         if '分派' in scheme and '每股派' not in scheme:
#             continue
#         # 跳过送转股
#         if '送股' in scheme or '转股' in scheme:
#             continue
#
#         div_hkd = parse_dividend_hkd(scheme)
#         if div_hkd is None or div_hkd <= 0:
#             continue
#
#         # 判断是否缺失：ex_date 晚于 Tushare 最后 adj_factor 变动日
#         is_missing = (last_adj_date is None) or (ex_date_str > last_adj_date)
#         if is_missing:
#             missing.append((ex_date_str, div_hkd))
#     return missing
#
#
# def apply_missing_adj(df: pd.DataFrame,
#                       missing: list[tuple[str, float]]) -> pd.DataFrame:
#     """对缺失的除权日，计算并应用 adj_factor 修正。
#
#     df 应包含 hk_daily_adj 原始列: trade_date, close, adj_factor 等
#     返回新增 adj_factor_corrected 列的 DataFrame。
#     """
#     df = df.copy()
#     df['adj_factor_corrected'] = df['adj_factor'].astype(float)
#     n = len(df)
#
#     for ex_date_str, div_hkd in missing:
#         # 找到除权日所在行（若当日非交易日则取其后的第一个交易日）
#         ex_dt = pd.Timestamp(ex_date_str)
#         pos = None
#         for i in range(n):
#             if df.iloc[i]['trade_date'] >= ex_dt:
#                 pos = i
#                 break
#         if pos is None:
#             continue
#
#         close_ex = float(df.iloc[pos]['close'])
#         if pd.isna(close_ex) or close_ex <= 0:
#             continue
#
#         # 前复权修正: 除权日之前的价格 × correction
#         # correction = close / (close + dividend)
#         correction = close_ex / (close_ex + div_hkd)
#
#         # 除权日之前 (0 ~ pos-1) 乘以 correction
#         if pos > 0:
#             col_idx = df.columns.get_loc('adj_factor_corrected')
#             df.iloc[:pos, col_idx] *= correction
#
#     return df
#
#
# def compute_adjusted_prices(df: pd.DataFrame) -> pd.DataFrame:
#     """用 adj_factor_corrected 生成前复权 OHLC 列 (close/high/low/open)"""
#     adj = df['adj_factor_corrected'].values
#     for col in ['close', 'high', 'low', 'open']:
#         if col in df.columns:
#             df[col] = (df[col] * adj).round(4)
#     return df
#
#
# # ============================================================
# # 唐奇安指标计算
# # ============================================================
# def compute_indicators(df: pd.DataFrame,ema_fast,ema_slow) -> pd.DataFrame:
#     df = df.copy().sort_values("trade_date").reset_index(drop=True)
#     df["TR"] = np.maximum(
#         df["high"] - df["low"],
#         np.maximum(
#             abs(df["high"] - df["close"].shift(1)),
#             abs(df["low"] - df["close"].shift(1)),
#         ),
#     )
#     df["ATR"] = df["TR"].rolling(ATR_PERIOD, min_periods=ATR_PERIOD).mean()
#     df["HH_Entry"] = df["high"].rolling(ENTRY_PERIOD).max().shift(1)
#     df["LL_Exit"] = df["low"].rolling(EXIT_PERIOD).min().shift(1)
#     df["EMA_Fast"] = df["close"].ewm(span=ema_fast, adjust=False).mean()
#     df["EMA_Slow"] = df["close"].ewm(span=ema_slow, adjust=False).mean()
#     return df
#
#
# # ============================================================
# # 加仓做多回测
# # ============================================================
# def backtest(df: pd.DataFrame, ema_fast,ema_slow,initial_capital: float = INITIAL_CAPITAL):
#     df = compute_indicators(df,ema_fast,ema_slow)
#     trades, capital, eq = [], initial_capital, [initial_capital]
#     pos = None
#     ep = eatr = slp = 0.0
#     cs = bs = ms = tc = 0
#     ed = None
#     mi = []
#
#     start_idx = max(ENTRY_PERIOD, ATR_PERIOD, EMA_SLOW) + 2
#
#     for i in range(start_idx, len(df)):
#         row = df.iloc[i]
#         cur = row["trade_date"]
#
#         if pos is None:
#             atr = row["ATR"]
#             if pd.isna(atr) or atr == 0:
#                 continue
#             sd = N_VALUE * atr
#             shares = int(capital * RISK_PERCENT / (sd * PER_SHARES))
#             if shares <= 0:
#                 continue
#             bs = shares
#             skip_offset = 1 if SKIP_FIRST_ADD else 0
#             ms = shares * (MAX_PYRAMID - skip_offset)
#
#             fv = row.get("EMA_Fast")
#             sv = row.get("EMA_Slow")
#             ema_ok = pd.notna(fv) and pd.notna(sv) and fv > sv
#
#             if row["close"] > row["HH_Entry"] and ema_ok:
#                 cost = row["close"] * shares * PER_SHARES
#                 if cost > capital * 0.8:
#                     continue
#                 pos = "long"
#                 ep = row["close"]
#                 eatr = atr
#                 slp = ep - sd
#                 cs = shares
#                 tc = cost
#                 ed = cur
#                 capital -= cost
#
#         else:
#             exit_price = None
#             exit_reason = ""
#             if row["close"] < slp:
#                 exit_price = row["close"]
#                 exit_reason = "止损(2×ATR)"
#             elif row["close"] < row["LL_Exit"]:
#                 exit_price = row["close"]
#                 exit_reason = "趋势反转(10日低点)"
#
#             if not exit_reason and eatr > 0:
#                 skip_offset = 1 if SKIP_FIRST_ADD else 0
#                 uh = cs // bs + skip_offset
#                 if row["close"] > (ep + uh * 0.5 * eatr):
#                     if cs < ms:
#                         ap = row["close"]
#                         slp = ap - N_VALUE * eatr
#                         ac = ap * bs * PER_SHARES
#                         if ac <= capital * 0.8:
#                             cs += bs
#                             tc += ac
#                             capital -= ac
#                             mi.append({"add_price": ap})
#
#             if exit_reason:
#                 sp = exit_price * cs * PER_SHARES
#                 pnl = sp - tc
#                 capital += sp
#                 trades.append({
#                     "entry_date": ed, "exit_date": cur,
#                     "entry_price": ep, "exit_price": exit_price,
#                     "shares": cs, "cost": tc,
#                     "pnl": pnl, "pnl_pct": pnl / tc if tc else 0,
#                     "reason": exit_reason, "adds": len(mi),
#                     "hold_days": (cur - ed).days,
#                 })
#                 eq.append(capital)
#                 pos = None
#                 ep = eatr = slp = 0.0
#                 cs = bs = ms = tc = 0
#                 ed = None
#                 mi = []
#
#     if pos is not None:
#         lp = df.iloc[-1]["close"]
#         ld = df.iloc[-1]["trade_date"]
#         sp = lp * cs * PER_SHARES
#         pnl = sp - tc
#         capital += sp
#         trades.append({
#             "entry_date": ed, "exit_date": ld,
#             "entry_price": ep, "exit_price": lp,
#             "shares": cs, "cost": tc,
#             "pnl": pnl, "pnl_pct": pnl / tc if tc else 0,
#             "reason": "强制平仓", "adds": len(mi),
#             "hold_days": (ld - ed).days,
#         })
#         eq.append(capital)
#
#     return trades, eq
#
#
# # ============================================================
# # 评估
# # ============================================================
# def evaluate(trades: list, equity_curve: list,
#              initial_capital: float = INITIAL_CAPITAL) -> dict:
#     if not trades:
#         return {
#             "return": 0, "wr": 0, "pf": 0, "dd": 0, "n": 0, "adds": 0,
#             "hold": 0, "avg_pnl": 0, "aw": 0, "al": 0,
#             "best": 0, "worst": 0, "sharpe": 0, "calmar": 0,
#             "consec": 0, "final": initial_capital,
#         }
#
#     wins = [t for t in trades if t["pnl"] > 0]
#     losses = [t for t in trades if t["pnl"] <= 0]
#
#     ret = (equity_curve[-1] / initial_capital - 1) * 100
#     wr = len(wins) / len(trades) * 100
#     ws = sum(t["pnl"] for t in wins)
#     ls = abs(sum(t["pnl"] for t in losses))
#     pf = ws / ls if ls > 0 else (float("inf") if ws > 0 else 0)
#
#     peak = equity_curve[0]
#     dd = 0.0
#     for v in equity_curve:
#         if v > peak:
#             peak = v
#         d = (peak - v) / peak
#         if d > dd:
#             dd = d
#     dd *= 100
#
#     pcts = [t["pnl_pct"] for t in trades]
#     mu, std = np.mean(pcts), np.std(pcts)
#     sharpe = mu / std if std > 0 else 0
#     calmar = ret / dd if dd > 0 else float("inf")
#
#     consec, cur_c = 0, 0
#     for t in trades:
#         if t["pnl"] <= 0:
#             cur_c += 1
#             consec = max(consec, cur_c)
#         else:
#             cur_c = 0
#
#     return {
#         "return": round(ret, 2),
#         "wr": round(wr, 1),
#         "pf": round(pf, 2),
#         "dd": round(dd, 2),
#         "n": len(trades),
#         "adds": sum(t["adds"] for t in trades),
#         "hold": round(np.mean([t["hold_days"] for t in trades]), 0),
#         "avg_pnl": round(sum(t["pnl"] for t in trades) / len(trades), 2),
#         "aw": round(np.mean([t["pnl"] for t in wins]), 2) if wins else 0,
#         "al": round(np.mean([t["pnl"] for t in losses]), 2) if losses else 0,
#         "best": round(max(pcts) * 100, 2),
#         "worst": round(min(pcts) * 100, 2),
#         "sharpe": round(sharpe, 2),
#         "calmar": round(calmar, 2),
#         "consec": consec,
#         "final": round(equity_curve[-1], 2),
#     }
#
#
# # ============================================================
# # 主流程
# # ============================================================
# def main():
#     token = os.environ.get("TUSHARE_TOKEN")
#     if not token:
#         print("❌ 请设置 TUSHARE_TOKEN")
#         return
#     pro = ts.pro_api(token)
#
#     # 获取上市日期 + 名称
#     if MARKET == "A":
#         sb = pro.stock_basic(list_status='L', fields='ts_code,name,list_date')
#         listing = {}
#         name_map = {}
#         for _, r in sb.iterrows():
#             listing[r["ts_code"]] = r["list_date"]
#             name_map[r["ts_code"]] = r["name"]
#     else:
#         hk = pro.hk_basic()
#         listing = {}
#         name_map = {}
#         for _, r in hk.iterrows():
#             listing[r["ts_code"]] = r["list_date"]
#             name_map[r["ts_code"]] = r["name"]
#
#     # 从 optimize_donchian_ema 获取流动性 TOP10 股票列表
#     currency = "CNY" if MARKET == "A" else "HKD"
#     market_label = "A股" if MARKET == "A" else "港股"
#     print("=" * 100)
#     print(f"🏆 近五年{market_label}日均成交额 TOP10")
#     print("=" * 100)
#     top10_stocks = fetch_top10_by_liquidity(pro, CANDIDATES, LIQ_START, END_DATE)
#     for i, (code, amt, _) in enumerate(top10_stocks, 1):
#         name = name_map.get(code, "")
#         print(f"  #{i} {code} {name}  日均成交额: {amt:>15,.0f} {currency}")
#     CODES = [code for code, _, _ in top10_stocks]
#     # 获取过滤器最优参数
#     summary = run_parameter_scan(top10_stocks)
#     best = summary[summary["ema_fast"] > 0].iloc[0]
#     EMA_FAST = best['ema_fast']
#     EMA_SLOW = best['ema_slow']
#     data_source = "hk_daily_adj + akshare分红补权" if MARKET == "HK" else "daily + adj_factor"
#     print(f"\n{'=' * 100}")
#     print(f"📊 唐奇安策略回测  EMA({EMA_FAST},{EMA_SLOW})  加仓做多  max_pyramid={MAX_PYRAMID}")
#     print(f"   数据源: {data_source} → 前复权价格")
#     print("=" * 100)
#
#     data_dir = os.path.join(os.path.dirname(__file__), f"{'a' if MARKET == 'A' else 'hk'}_daily_data")
#     os.makedirs(data_dir, exist_ok=True)
#
#
#     rows = []
#     for code in CODES:
#         sdate = listing.get(code, "20210601")
#         name = name_map.get(code, "")
#         try:
#             if MARKET == "HK":
#                 # ---- 港股: hk_daily_adj + akshare补权 ----
#                 df = pro.hk_daily_adj(ts_code=code, start_date=sdate, end_date=END_DATE)
#                 if df is None or df.empty:
#                     continue
#                 df["trade_date"] = pd.to_datetime(df["trade_date"])
#                 df = df.sort_values("trade_date").reset_index(drop=True)
#
#                 missing = find_missing_dividends(pro, code, df)
#                 if missing:
#                     print(f"     ⚠️ adj_factor缺失 {len(missing)} 笔分红:")
#                     for ex_d, amt in missing:
#                         print(f"        除净日={ex_d}  HKD {amt:.4f}/股")
#                     df = apply_missing_adj(df, missing)
#                 else:
#                     df['adj_factor_corrected'] = df['adj_factor']
#                 df = compute_adjusted_prices(df)
#                 adj_missing = len(missing)
#             else:
#                 # ---- A股: daily + adj_factor ----
#                 df = pro.daily(ts_code=code, start_date=sdate, end_date=END_DATE)
#                 if df is None or df.empty:
#                     continue
#                 df["trade_date"] = pd.to_datetime(df["trade_date"])
#                 df = df.sort_values("trade_date").reset_index(drop=True)
#
#                 adj = pro.adj_factor(ts_code=code)
#                 adj["trade_date"] = pd.to_datetime(adj["trade_date"])
#                 # merge + 后向填充 adj_factor
#                 df = df.merge(adj[['trade_date', 'adj_factor']], on='trade_date', how='left')
#                 last = 1.0
#                 filled = []
#                 for i in range(len(df) - 1, -1, -1):
#                     v = df['adj_factor'].iloc[i]
#                     if pd.notna(v): last = float(v)
#                     filled.append(last)
#                 filled.reverse()
#                 df['adj_factor'] = filled
#                 last_adj = df['adj_factor'].iloc[-1]
#                 for col in ['close', 'high', 'low', 'open']:
#                     if col in df.columns:
#                         df[col] = (df[col] * df['adj_factor']/last_adj).round(4)
#                 adj_missing = 0  # A股Tushare复权完善，无需补权
#
#             # ---- 保存日线到 Excel ----
#             excel_path = os.path.join(data_dir, f"{code}.xlsx")
#             out_cols = ['trade_date', 'open', 'high', 'low', 'close', 'vol', 'amount']
#             df[out_cols].to_excel(excel_path, index=False)
#
#             # ---- 回测 ----
#             trades, eq = backtest(df, EMA_FAST, EMA_SLOW)
#             m = evaluate(trades, eq)
#             m["stock"] = f"{code} {name}"
#             m["code"] = code
#             m["sdate"] = sdate
#             m["days"] = len(df)
#             m["adj_missing"] = adj_missing
#             m["data_file"] = excel_path
#             rows.append(m)
#
#             adj_str = f"补{adj_missing}笔" if adj_missing else "完整"
#             print(f"  ✅ {code} {name:<8}: {sdate}→今 {len(df)}天  "
#                   f"adj:{adj_str}  交易{m['n']}笔  回报{m['return']:+.1f}%")
#
#         except Exception as e:
#             print(f"  ❌ {code} {name}: {e}")
#             import traceback; traceback.print_exc()
#
#     # ---- 综合评分 ----
#     if not rows:
#         print("❌ 没有成功获取任何数据")
#         return
#
#     rdf = pd.DataFrame(rows)
#
#     def _norm(s):
#         mn, mx = s.min(), s.max()
#         return (s - mn) / (mx - mn) if mx > mn else pd.Series(0.5, index=s.index)
#
#     rdf["score_return"] = _norm(rdf["return"])
#     rdf["score_pf"] = _norm(rdf["pf"])
#     rdf["score_dd"] = _norm(1 - rdf["dd"] / 100)
#     rdf["score_sharpe"] = _norm(rdf["sharpe"])
#     rdf["score_wr"] = _norm(rdf["wr"])
#     rdf["score_calmar"] = _norm(rdf["calmar"])
#
#     rdf["fitness"] = (
#         rdf["score_return"] * 0.40 +
#         rdf["score_pf"] * 0.20 +
#         rdf["score_dd"] * 0.15 +
#         rdf["score_sharpe"] * 0.10 +
#         rdf["score_wr"] * 0.10 +
#         rdf["score_calmar"] * 0.05
#     )
#
#     rdf = rdf.sort_values("fitness", ascending=False).reset_index(drop=True)
#     rdf["rank"] = range(1, len(rdf) + 1)
#
#     def grade(s):
#         if s >= 0.7: return "⭐⭐⭐⭐⭐ 极佳"
#         elif s >= 0.55: return "⭐⭐⭐⭐ 优秀"
#         elif s >= 0.4: return "⭐⭐⭐ 良好"
#         elif s >= 0.25: return "⭐⭐ 一般"
#         return "⭐ 不适合"
#
#     rdf["grade"] = rdf["fitness"].apply(grade)
#
#     # ---- 输出 ----
#     cols = ["rank", "stock", "fitness", "grade", "adj_missing", "sdate", "days",
#             "return", "wr", "pf", "dd", "calmar", "sharpe", "n", "adds",
#             "aw", "al", "best", "worst", "consec", "hold", "final"]
#     print("\n" + "=" * 130)
#     print("🏆 唐奇安策略适配度排名 (前复权)")
#     print("=" * 130)
#     print(rdf[cols].to_string(index=False))
#
#     print("\n🎯 结论:")
#     print("  ✅ 最适合 (建议重点配置):")
#     for _, r in rdf.head(3).iterrows():
#         print(f"     #{int(r['rank'])} {r['stock']:<22} "
#               f"适配度{r['fitness']:.3f}  回报{r['return']:+.1f}%  "
#               f"胜率{r['wr']}%  回撤{r['dd']}%  盈亏比{r['pf']:.2f}")
#     print("  ⚠️ 不适合 (建议回避):")
#     for _, r in rdf.tail(3).iterrows():
#         print(f"     #{int(r['rank'])} {r['stock']:<22} "
#               f"适配度{r['fitness']:.3f}  回报{r['return']:+.1f}%  "
#               f"胜率{r['wr']}%  回撤{r['dd']}%  盈亏比{r['pf']:.2f}")
#
#     rdf.to_csv(OUTPUT, index=False)
#     print(f"\n✅ 已保存 {OUTPUT}")
#
#
# if __name__ == "__main__":
#     main()
