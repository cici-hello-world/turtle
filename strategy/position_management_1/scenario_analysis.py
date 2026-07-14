#!/usr/bin/env python3
"""理想汽车(02015.HK) 行情形态识别 & 未来情景概率估算

基于 2025-10 至今的实际走势，与 8 种模拟行情做定量匹配，
结合宏观环境估算未来 6 个月各情景的概率。

用法：
  python scenario_analysis.py           # 分析理想汽车
  python scenario_analysis.py 03690.HK  # 分析其他标的
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════
# 1. 8种模拟行情定义 (与 simulate_market_scenarios.py 保持一致)
# ══════════════════════════════════════════════════════════════════════

SCENARIOS = {
    "单边上涨": {
        "slope": 0.008, "osc_amp": 0.00, "osc_period": 60, "noise": 0.012,
        "desc": "日均+0.8%, 低噪声, 强劲趋势向上",
    },
    "单边下跌": {
        "slope": -0.008, "osc_amp": 0.00, "osc_period": 60, "noise": 0.012,
        "desc": "日均-0.8%, 低噪声, 崩溃式下跌",
    },
    "震荡向上": {
        "slope": 0.003, "osc_amp": 0.06, "osc_period": 14, "noise": 0.016,
        "desc": "日均+0.3%, ±6%振幅, 趋势向上但摆动明显",
    },
    "震荡向下": {
        "slope": -0.002, "osc_amp": 0.08, "osc_period": 12, "noise": 0.017,
        "desc": "日均-0.2%, ±8%振幅, 重心下移伴大摆动",
    },
    "横盘震荡": {
        "slope": 0.0, "osc_amp": 0.06, "osc_period": 20, "noise": 0.014,
        "desc": "零趋势, ±6%箱体, 方向不明",
    },
    "V形反转": {
        "slope": 0.0, "osc_amp": 0.00, "osc_period": 60, "noise": 0.014,
        "desc": "前半跌后半涨, 触底反弹",
    },
    "倒V反转": {
        "slope": 0.0, "osc_amp": 0.00, "osc_period": 60, "noise": 0.014,
        "desc": "前半涨后半跌, 冲高回落",
    },
    "缓慢阴跌": {
        "slope": -0.0035, "osc_amp": 0.03, "osc_period": 10, "noise": 0.012,
        "desc": "日均-0.35%, ±3%小反弹, 钝刀子割肉",
    },
}


def load_data(ts_code: str) -> pd.DataFrame:
    """加载数据并计算基础指标"""
    market = "HK" if ts_code.split(".")[1] == "HK" else "A"
    cache_dir = Path(__file__).parent.parent.parent / "data_cache" / market
    path = cache_dir / f"{ts_code}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"数据文件不存在: {path}")

    df = pd.read_parquet(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df["Return"] = df["Close"].pct_change()
    return df


def match_scenarios(df: pd.DataFrame, start_date: str = "2025-10-01") -> dict:
    """将近期走势与8种场景做定量匹配，返回各场景得分"""
    mask = df["Date"] >= start_date
    recent = df[mask].copy()

    if len(recent) < 20:
        return {}

    close = recent["Close"].values
    rets = recent["Return"].dropna().values
    n = len(close)

    # ── 实际数据特征 ──
    actual_slope = (close[-1] / close[0]) ** (1 / n) - 1  # 几何日均变化率
    actual_vol = np.std(rets)
    actual_dd = (close[-1] / np.maximum.accumulate(close).max() - 1) * 100
    actual_up_ratio = (rets > 0).sum() / len(rets)
    total_chg = (close[-1] / close[0] - 1) * 100

    features = {
        "n_days": n,
        "total_chg": total_chg,
        "daily_slope": actual_slope * 100,
        "daily_vol": actual_vol * 100,
        "max_dd": actual_dd,
        "up_ratio": actual_up_ratio * 100,
        "start_price": close[0],
        "end_price": close[-1],
        "high": close.max(),
        "low": close.min(),
    }

    scores = {}
    for name, params in SCENARIOS.items():
        # 方向匹配
        actual_dir = 1 if actual_slope > 0.0005 else (-1 if actual_slope < -0.0005 else 0)
        scenario_dir = 1 if params["slope"] > 0.0005 else (-1 if params["slope"] < -0.0005 else 0)
        dir_score = 1.0 if actual_dir == scenario_dir else (0.5 if scenario_dir == 0 else 0.0)

        # 日趋势强度匹配
        slope_diff = abs(abs(actual_slope) - abs(params["slope"]))
        slope_score = max(0.0, 1.0 - slope_diff / 0.008)

        # 波动率匹配
        vol_diff = abs(actual_vol - params["noise"])
        vol_score = max(0.0, 1.0 - vol_diff / 0.020)

        # 总变化幅度匹配 (用趋势项累积 ≈ slope × n)
        expected_total = ((1 + params["slope"]) ** n - 1) * 100
        total_diff = abs(total_chg - expected_total)
        total_score = max(0.0, 1.0 - total_diff / 80.0)

        # 上涨日比例匹配
        expected_up = 0.5 + params["slope"] / 0.02
        expected_up = max(0.2, min(0.8, expected_up))
        up_diff = abs(actual_up_ratio - expected_up)
        up_score = max(0.0, 1.0 - up_diff / 0.30)

        # 综合得分 (加权)
        composite = (
            dir_score * 0.30
            + slope_score * 0.20
            + vol_score * 0.15
            + total_score * 0.20
            + up_score * 0.15
        )

        scores[name] = {
            "composite": round(composite, 3),
            "dir": round(dir_score, 2),
            "slope": round(slope_score, 2),
            "vol": round(vol_score, 2),
            "total": round(total_score, 2),
            "up": round(up_score, 2),
        }

    return {"features": features, "scores": scores}


# ══════════════════════════════════════════════════════════════════════
# 个股宏观背景 (用于概率估算的定性调整)
# ══════════════════════════════════════════════════════════════════════

STOCK_MACRO = {
    "02015.HK": {  # 理想汽车
        "name": "理想汽车",
        "sector": "新能源车",
        "slow_grind": "价格战白热化+消费降级，高端SUV需求走弱",
        "volatile_down": "间歇性政策利好但新能源车竞争格局不支撑反转",
        "sideways": "估值已大幅压缩，技术面超卖，多空暂时平衡",
        "v_reversal": "L系列换代成功/MEGA改款/出口突破驱动估值修复",
        "volatile_up": "行业整合+以旧换新补贴+利润率修复",
        "crash": "流动性危机/大规模召回/毛利率暴雷",
        "inv_v": "短暂反弹后竞争压力再次显现",
        "up": "爆款新车+行业拐点+政策组合拳",
    },
    "03690.HK": {  # 美团
        "name": "美团",
        "sector": "本地生活/外卖",
        "slow_grind": "到店竞争胶着+外卖增速放缓+社区团购亏损收窄但未盈利",
        "volatile_down": "抖音冲击到店份额，外卖客单价承压，重心下移",
        "sideways": "估值泡沫消化，外卖基本盘稳固，多空均衡",
        "v_reversal": "到店份额回稳/海外外卖突破/新业务减亏超预期",
        "volatile_up": "外卖UE改善+到店份额回升+新业务逐步减亏",
        "crash": "反垄断重罚/骑手社保成本暴增/抖音全面超越到店",
        "inv_v": "季报超预期短期拉升后竞争压力再现",
        "up": "外卖+到店双轮驱动+海外扩张+AI赋能骑手调度",
    },
    "06181.HK": {  # 老铺黄金
        "name": "老铺黄金",
        "sector": "黄金珠宝零售",
        "slow_grind": "金价高位抑制消费+可选消费降级，终端动销放缓",
        "volatile_down": "金价波动+开店不及预期，间歇性反弹后重心下移",
        "sideways": "品牌溢价+黄金避险属性支撑估值，多空平衡",
        "v_reversal": "金价回落刺激消费/古法金工艺出圈/加盟扩张加速",
        "volatile_up": "消费回暖+金价企稳+品牌升级驱动毛利率提升",
        "crash": "金价暴跌+库存减值+加盟商大面积退网",
        "inv_v": "短期事件驱动拉升后金价反转回落",
        "up": "黄金大牛市+古法金品类爆发+渠道下沉共振",
    },
    "09992.HK": {  # 泡泡玛特
        "name": "泡泡玛特",
        "sector": "潮玩/IP经济",
        "slow_grind": "IP热度周期性退潮+消费降级挤压可选消费品",
        "volatile_down": "海外扩张受阻或新IP不及预期，重心缓慢下移",
        "sideways": "估值回归合理区间，海外+国内双轮驱动多空平衡",
        "v_reversal": "爆款IP引爆情绪/海外门店超预期/盲盒监管放松",
        "volatile_up": "海外市场持续高增+IP矩阵扩展+利润率改善",
        "crash": "核心IP热度崩塌/海外门店大面积亏损/库存暴雷",
        "inv_v": "新IP短期拉升后热度回落，竞争加剧",
        "up": "全球IP平台化+海外爆发+消费回暖共振",
    },
}

# 通用模板 (未收录的标的)
_GENERIC_MACRO = {
    "slow_grind": "行业景气下行+竞争加剧，缺乏新增长点",
    "volatile_down": "间歇性利好但基本面不支撑反转，重心下移",
    "sideways": "估值回落至合理区间，多空暂时平衡",
    "v_reversal": "政策利好/业务突破驱动估值修复",
    "volatile_up": "行业回暖+利润率修复，缓慢回升",
    "crash": "流动性危机/经营暴雷/行业黑天鹅",
    "inv_v": "短暂反弹后压力再次显现",
    "up": "行业拐点+公司突破+政策组合拳共振",
}


def _get_macro(ts_code: str) -> dict:
    """获取个股特定的宏观背景说明"""
    if ts_code in STOCK_MACRO:
        return STOCK_MACRO[ts_code]
    # 尝试模糊匹配
    for key in STOCK_MACRO:
        if key.startswith(ts_code[:5]):
            return STOCK_MACRO[key]
    return {"name": ts_code, "sector": "通用", **_GENERIC_MACRO}


def _scenario_signal_score(name: str, params: dict, features: dict) -> float:
    """基于当前实时行情信号，评估某情景作为未来走势的合理程度。

    不是历史模式匹配，而是从"当前位置 + 情景方向 + 波动特征"的契合度打分。
    """
    slope = features["daily_slope"] / 100   # 转回小数
    vol = features["daily_vol"] / 100
    up_ratio = features["up_ratio"] / 100
    max_dd = features["max_dd"] / 100
    total_chg = features["total_chg"] / 100

    score = 1.0  # 基础分

    # ── 1. 方向契合度 ──
    # 情景的未来方向 vs 当前动量：当前方向与情景终点方向越一致，延续该路径越自然
    scenario_slope = params["slope"]
    if scenario_slope > 0.0005:       # 情景终点向上
        if slope > 0.0005:
            score *= 1.3               # 顺势：当前向上 → 未来继续向上，自然
        elif slope < -0.0005:
            score *= 0.5               # 逆势：当前下跌 → 要翻转为上涨，需更大能量
        else:
            score *= 0.85
    elif scenario_slope < -0.0005:    # 情景终点向下
        if slope < -0.0005:
            score *= 1.3
        elif slope > 0.0005:
            score *= 0.5
        else:
            score *= 0.85
    else:                              # 情景终点横盘
        score *= 1.1 if abs(slope) < 0.001 else 0.8

    # ── 2. 波动率契合度 ──
    # 情景的典型波动 vs 当前波动：当前波动越接近情景特征，越可能延续
    scenario_noise = params["noise"]
    vol_ratio = vol / scenario_noise if scenario_noise > 0 else 1.0
    if 0.5 <= vol_ratio <= 2.0:
        score *= 1.15                  # 波动率匹配
    elif vol_ratio > 3.0:
        score *= 0.7                   # 当前波动远超情景特征

    # ── 3. V形/倒V 的特殊前置条件 ──
    if name == "V形反转":
        # V形需要前期已发生较大回撤（才有反弹空间）
        if max_dd < -0.15:
            score *= 1.5               # 深度回调后 V 形反转更可信
        elif max_dd < -0.08:
            score *= 1.15
        else:
            score *= 0.6               # 没怎么跌，V形前提不成立
    if name == "倒V反转":
        # 倒V需要前期已有较大涨幅（才有回落空间）
        if total_chg > 0.20:
            score *= 1.5
        elif total_chg > 0.10:
            score *= 1.15
        else:
            score *= 0.6

    # ── 4. 单边趋势需要低波动配合 ──
    if name in ("单边上涨", "单边下跌"):
        if vol < 0.018:
            score *= 1.2               # 低波动有利于趋势延续
        elif vol > 0.035:
            score *= 0.7               # 高波动不利于单边趋势

    # ── 5. 震荡情景需要高波动 ──
    if name in ("震荡向上", "震荡向下"):
        if vol > 0.025:
            score *= 1.15
        elif vol < 0.014:
            score *= 0.8

    # ── 6. 横盘震荡：当前波动适中 + 方向不明 ──
    if name == "横盘震荡":
        if abs(slope) < 0.0008 and 0.012 < vol < 0.025:
            score *= 1.25
        elif abs(slope) > 0.002:
            score *= 0.7

    # ── 7. 缓慢阴跌：方向略偏下 + 低波动 ──
    if name == "缓慢阴跌":
        if slope < 0 and vol < 0.02:
            score *= 1.2
        elif slope > 0.001:
            score *= 0.7

    return score


def estimate_probabilities(scores: dict, ts_code: str = "",
                           features: dict = None) -> dict:
    """基于当前行情走势 + 宏观环境实时推测未来6个月各情景概率。

    不使用历史模式匹配。信号来自 features（近期的趋势、波动、回撤），
    宏观来自 STOCK_MACRO 的个股定性背景。
    """
    macro = _get_macro(ts_code)
    sorted_by_match = sorted(scores.items(), key=lambda x: x[1]["composite"], reverse=True)

    # ── 从实时信号计算各情景得分 ──
    if features:
        raw_scores = {}
        for name, params in SCENARIOS.items():
            raw_scores[name] = _scenario_signal_score(name, params, features)

        # ── 宏观调整：根据个股背景微调 ±20% ──
        macro_bias = {
            "缓慢阴跌": 1.05, "震荡向下": 1.05, "横盘震荡": 1.0,
            "V形反转": 0.95, "震荡向上": 1.0, "单边下跌": 0.95,
            "倒V反转": 0.95, "单边上涨": 0.95,
        }
        for name in raw_scores:
            raw_scores[name] *= macro_bias.get(name, 1.0)

        # ── softmax 转概率 ──
        temp = 0.3
        exp_scores = {k: np.exp(v / temp) for k, v in raw_scores.items()}
        total = sum(exp_scores.values())
        probs = {k: round(v / total, 4) for k, v in exp_scores.items()}
    else:
        # 无 features 时退化为等权
        probs = {k: round(1.0 / len(SCENARIOS), 4) for k in SCENARIOS}

    probs = dict(sorted(probs.items(), key=lambda x: x[1], reverse=True))

    note_key = {
        "缓慢阴跌": macro.get("slow_grind", ""),
        "震荡向下": macro.get("volatile_down", ""),
        "横盘震荡": macro.get("sideways", ""),
        "V形反转": macro.get("v_reversal", ""),
        "震荡向上": macro.get("volatile_up", ""),
        "单边下跌": macro.get("crash", ""),
        "倒V反转": macro.get("inv_v", ""),
        "单边上涨": macro.get("up", ""),
    }

    down = sum(v for k, v in probs.items() if "跌" in k or "倒V" in k)
    side = sum(v for k, v in probs.items() if "震荡" in k or "横盘" in k)
    up = sum(v for k, v in probs.items() if "涨" in k or ("V" in k and "倒" not in k))

    return {
        "probs": probs,
        "macro_notes": note_key,
        "stock_name": macro.get("name", ts_code),
        "sector": macro.get("sector", ""),
        "summary": {"down": down, "sideways": side, "up": up},
        "top_match": sorted_by_match[0][0],
        "second_match": sorted_by_match[1][0] if len(sorted_by_match) > 1 else None,
    }


def print_report(ts_code: str, result: dict, estimates: dict):
    """打印完整分析报告"""
    feat = result["features"]
    scores = result["scores"]

    stock_name = estimates.get("stock_name", ts_code)
    sector = estimates.get("sector", "")
    sector_str = f"  |  {sector}" if sector else ""

    print("=" * 75)
    print(f"  {ts_code} {stock_name} 行情形态识别 & 情景概率估算{sector_str}")
    print(f"  分析区间: 2025-10-01 → 最新 ({feat['n_days']}个交易日)")
    print("=" * 75)

    # ── 实际走势特征 ──
    print(f"\n📈 实际走势特征:")
    print(f"  起始 ¥{feat['start_price']:.1f} → 最新 ¥{feat['end_price']:.1f}"
          f"  ({feat['total_chg']:+.1f}%)")
    print(f"  最高 ¥{feat['high']:.1f}  最低 ¥{feat['low']:.1f}"
          f"  最大回撤 {feat['max_dd']:.1f}%")
    print(f"  日趋势 {feat['daily_slope']:+.3f}%  日波动 {feat['daily_vol']:.2f}%"
          f"  涨日比 {feat['up_ratio']:.0f}%")

    # ── 场景匹配 ──
    print(f"\n🎯 与8种模拟行情的定量匹配:")
    print(f"  {'场景':<10} {'综合':>6} {'方向':>5} {'斜率':>5} {'波动':>5}"
          f" {'总变化':>5} {'涨日比':>5}  {'匹配度'}")
    print(f"  {'─' * 55}")
    sorted_scores = sorted(scores.items(), key=lambda x: x[1]["composite"], reverse=True)
    for name, s in sorted_scores:
        bar = "█" * int(s["composite"] * 25)
        print(f"  {name:<10} {s['composite']:>5.2f} {s['dir']:>5.2f} {s['slope']:>5.2f}"
              f" {s['vol']:>5.2f} {s['total']:>5.2f} {s['up']:>5.2f}  {bar}")

    top1, top2 = estimates["top_match"], estimates["second_match"]
    s1 = SCENARIOS[top1]
    s2 = SCENARIOS[top2]
    print(f"\n  🥇 最佳匹配: {top1} — {s1['desc']}")
    print(f"  🥈 次佳匹配: {top2} — {s2['desc']}")

    # ── 未来概率 ──
    probs = estimates["probs"]
    notes = estimates["macro_notes"]
    summary = estimates["summary"]

    print(f"\n🔮 未来6个月情景概率:")
    print(f"  {'情景':<10} {'概率':>6}  {'说明'}")
    print(f"  {'─' * 55}")
    for name, prob in probs.items():
        bar = "█" * int(prob * 40)
        arrow = ("📉" if "跌" in name or "倒V" in name
                 else ("📈" if "涨" in name or ("V" in name and "倒" not in name)
                       else "📊"))
        print(f"  {arrow} {name:<8} {prob:>5.0%}  {bar}  {notes.get(name, '')}")

    print(f"\n  ─────────────")
    print(f"  下行风险: {summary['down']:.0%}  "
          f"震荡横盘: {summary['sideways']:.0%}  "
          f"上行机会: {summary['up']:.0%}")

    print(f"\n{'=' * 75}")


# ══════════════════════════════════════════════════════════════════════
# 历史区间匹配 (找到与未来最大概率走势相同的历史时段)
# ══════════════════════════════════════════════════════════════════════

def _scenario_target(estimates: dict) -> dict:
    """根据未来概率加权，合成"最可能走势"的目标特征。

    返回: daily_slope(小数), vol(%), up_ratio(%), total_chg_6m(%), label
    """
    probs = estimates["probs"]
    w_slope, w_vol, w_up, w_chg = 0.0, 0.0, 0.0, 0.0
    total_w = 0.0

    for name, prob in probs.items():
        params = SCENARIOS.get(name)
        if params is None:
            continue
        w = prob
        total_w += w
        # 日趋势 → 6个月(126天)总变化
        n_6m = 126
        chg_6m = ((1 + params["slope"]) ** n_6m - 1) * 100
        # 上涨日比例估算
        up_est = 50 + params["slope"] * 100 * 0.6  # 粗略: 斜率越大涨日越多

        w_slope += params["slope"] * w
        w_vol += params["noise"] * 100 * w
        w_up += up_est * w
        w_chg += chg_6m * w

    return {
        "daily_slope": w_slope / total_w if total_w > 0 else 0,
        "vol": w_vol / total_w if total_w > 0 else 0,
        "up_ratio": w_up / total_w if total_w > 0 else 50,
        "total_chg_6m": w_chg / total_w if total_w > 0 else 0,
    }


def find_historical_match(df: pd.DataFrame, estimates: dict,
                          min_days: int = 80, max_days: int = 200,
                          top_n: int = 15,
                          exclude_after: str = None) -> list:
    """找到历史中与未来最大概率走势最相似的时间段。

    用概率加权合成目标特征，而非匹配当前走势。
    exclude_after: 排除此日期之后的区间 (避免当前周期自我匹配)
    """
    close = df["Close"].values
    dates = df["Date"]
    step = 10
    exclude_cutoff = pd.Timestamp(exclude_after) if exclude_after else None
    results = []

    target = _scenario_target(estimates)
    t_slope = target["daily_slope"]
    t_vol = target["vol"]
    t_up = target["up_ratio"]
    t_chg = target["total_chg_6m"]

    for start in range(0, len(df) - min_days, step):
        for end in range(start + min_days, min(start + max_days, len(df)), step):
            if exclude_cutoff and dates.iloc[end - 1] >= exclude_cutoff:
                continue

            win_c = close[start:end]
            n = len(win_c)
            if n < min_days:
                continue

            total_chg = (win_c[-1] / win_c[0] - 1) * 100
            if abs(total_chg - t_chg) > 40:  # 总变化差太远
                continue

            rets = np.diff(win_c) / win_c[:-1]
            daily_slope = (win_c[-1] / win_c[0]) ** (1 / n) - 1
            vol = np.std(rets) * 100
            up_ratio = (rets > 0).sum() / len(rets) * 100
            peak = np.maximum.accumulate(win_c)
            max_dd = (win_c / peak - 1).min() * 100

            # 后半段加速下跌?
            half = n // 2
            s1 = (win_c[half] / win_c[0]) ** (1 / half) - 1
            s2 = (win_c[-1] / win_c[half]) ** (1 / (n - half)) - 1
            accel = s2 < s1 * 0.7

            # ── 与"最可能未来走势"的相似度 ──
            slope_sim = max(0.0, 1.0 - abs(daily_slope - t_slope) / 0.004)
            vol_sim = max(0.0, 1.0 - abs(vol - t_vol) / 3.0)
            up_sim = max(0.0, 1.0 - abs(up_ratio - t_up) / 20.0)
            chg_sim = max(0.0, 1.0 - abs(total_chg - t_chg) / 35.0)
            accel_bonus = 0.03 if accel else 0.0

            score = (slope_sim * 0.30 + vol_sim * 0.20
                     + up_sim * 0.10 + chg_sim * 0.40) + accel_bonus

            results.append({
                "start": dates.iloc[start],
                "end": dates.iloc[end - 1],
                "days": n,
                "total_chg": total_chg,
                "daily_slope": daily_slope * 100,
                "vol": vol,
                "up_ratio": up_ratio,
                "max_dd": max_dd,
                "accel": accel,
                "score": round(score, 4),
                "c_start": win_c[0],
                "c_end": win_c[-1],
                "c_min": win_c.min(),
                "c_max": win_c.max(),
            })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_n]


def print_historical_report(ts_code: str, df: pd.DataFrame,
                            estimates: dict, matches: list):
    """打印历史区间匹配报告 — 找到与未来最大概率走势最相似的历史时段"""
    target = _scenario_target(estimates)
    top_probs = list(estimates["probs"].items())[:3]

    print("=" * 95)
    print(f"  {ts_code} 历史区间匹配 — 找到与未来6个月最可能走势相同的历史时段")
    print(f"  未来概率加权目标: 日趋势{target['daily_slope']*100:+.3f}%  "
          f"波动{target['vol']:.2f}%  涨日比{target['up_ratio']:.0f}%  "
          f"6月变化{target['total_chg_6m']:+.1f}%")
    print(f"  驱动场景: " + " · ".join(f"{n}({p:.0%})" for n, p in top_probs))
    print("=" * 95)

    # ── TOP匹配 ──
    print(f"\n📊 历史最相似区间 TOP {len(matches)}:")
    print(f"  {'#':<3} {'区间':<24} {'天':>4} {'总跌幅':>8} {'日趋势':>8} "
          f"{'波动':>7} {'涨日%':>6} {'回撤':>7} {'加速':>4} {'得分':>6}")
    print(f"  {'─' * 85}")
    for i, r in enumerate(matches):
        acc = "✓" if r["accel"] else ""
        print(f"  {i+1:<3} {r['start'].strftime('%Y-%m-%d')}→{r['end'].strftime('%Y-%m-%d')} "
              f"{r['days']:>4} {r['total_chg']:>+7.1f}% {r['daily_slope']:>+7.3f}% "
              f"{r['vol']:>6.2f}% {r['up_ratio']:>5.0f}% {r['max_dd']:>+6.1f}% "
              f"{acc:>4} {r['score']:>6.4f}")

    # ── 最佳匹配详细月度 ──
    best = matches[0]
    mask = (df["Date"] >= best["start"]) & (df["Date"] <= best["end"])
    win = df[mask].copy()
    win["month"] = win["Date"].dt.to_period("M")
    monthly = win.groupby("month").agg(
        Close=("Close", "last"), High=("High", "max"), Low=("Low", "min"))
    monthly["chg"] = monthly["Close"].pct_change() * 100

    print(f"\n🏆 最佳匹配: {best['start'].strftime('%Y-%m-%d')} → "
          f"{best['end'].strftime('%Y-%m-%d')} ({best['days']}天)")
    print(f"   价格: ¥{best['c_start']:.1f} → ¥{best['c_end']:.1f} "
          f"({best['total_chg']:+.1f}%)")
    print(f"   日趋势: {best['daily_slope']:+.3f}%  波动: {best['vol']:.2f}%  "
          f"涨日比: {best['up_ratio']:.0f}%  最大回撤: {best['max_dd']:.1f}%")
    print(f"\n   月度表现:")
    phase = ""
    for m, r in monthly.iterrows():
        chg_str = f"{r['chg']:+.1f}%" if not pd.isna(r['chg']) else "起始"
        print(f"     {str(m):<8} ¥{r['Close']:>7.1f} {chg_str:>8}  "
              f"H¥{r['High']:>7.1f} L¥{r['Low']:>7.1f}  {phase}")

    # ── 后续走势 (历史能不能告诉我们什么) ──
    print(f"\n📈 历史后续走势 ({best['end'].strftime('%Y-%m')} → +6个月):")
    end_date = best["end"]
    future_end = end_date + pd.DateOffset(months=6)
    mask_future = (df["Date"] > end_date) & (df["Date"] <= future_end)
    future = df[mask_future].copy()
    future_ret = None

    if len(future) > 0:
        future["month"] = future["Date"].dt.to_period("M")
        f_monthly = future.groupby("month").agg(
            Close=("Close", "last"), High=("High", "max"), Low=("Low", "min"))
        f_monthly["chg"] = f_monthly["Close"].pct_change() * 100

        fc = future["Close"].values
        f_ret = (fc[-1] / fc[0] - 1) * 100
        f_min_ret = (fc.min() / fc[0] - 1) * 100
        f_max_ret = (fc.max() / fc[0] - 1) * 100

        print(f"   最近价: ¥{fc[0]:.1f} → ¥{fc[-1]:.1f} ({f_ret:+.1f}%)")
        print(f"   最低触及: ¥{fc.min():.1f} ({f_min_ret:+.1f}%)  "
              f"最高触及: ¥{fc.max():.1f} ({f_max_ret:+.1f}%)")
        print(f"\n   月度:")
        for m, r in f_monthly.iterrows():
            chg_str = f"{r['chg']:+.1f}%" if not pd.isna(r['chg']) else "起始"
            print(f"     {str(m):<8} ¥{r['Close']:>7.1f} {chg_str:>8}  "
                  f"H¥{r['High']:>7.1f} L¥{r['Low']:>7.1f}")
    else:
        print("   (该区间已是最近数据，无后续)")

    # ── 历史对照 ──
    print(f"\n  ═══════════════════════════════")
    print(f"  未来6个月最可能走势 → 历史对照")
    print(f"  ═══════════════════════════════")
    top_name = list(estimates["probs"].keys())[0]
    print(f"  主导场景: {top_name} ({estimates['probs'][top_name]:.0%})")
    print(f"  概率加权: 日趋势{target['daily_slope']*100:+.2f}%  "
          f"波动{target['vol']:.1f}%  6月变化{target['total_chg_6m']:+.0f}%")
    print(f"  历史最佳匹配 {best['start'].strftime('%Y-%m')}→{best['end'].strftime('%Y-%m')} "
          f"({best['days']}天 {best['total_chg']:+.0f}%)  "
          f"日趋势{best['daily_slope']:+.2f}%  波动{best['vol']:.1f}%")
    if future_ret is not None:
        print(f"  该段后续6月: {'反弹' if future_ret > 0 else '续跌' if future_ret < 0 else '横盘'}"
              f" ({future_ret:+.0f}%)")


class _Tee:
    """同时输出到终端和文件。"""
    def __init__(self, filepath: str):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def main():
    ts_code = sys.argv[1] if len(sys.argv) > 1 else "02015.HK"
    start_date = sys.argv[2] if len(sys.argv) > 2 else "2025-10-01"
    mode = sys.argv[3] if len(sys.argv) > 3 else "full"


    # ── 输出文件 ──
    out_dir = Path(__file__).parent.parent.parent / "result"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"scenario_analysis_{ts_code}_{pd.Timestamp.today().strftime('%Y%m%d')}.txt"
    tee = _Tee(str(report_path))
    original_stdout = sys.stdout
    sys.stdout = tee

    try:
        df = load_data(ts_code)

        # ── 形态匹配 + 概率估算 ──
        if mode in ("full", "scenario"):
            result = match_scenarios(df, start_date)
            if result:
                estimates = estimate_probabilities(result["scores"], ts_code,
                                                   features=result["features"])
                print_report(ts_code, result, estimates)

        # ── 历史区间匹配 ──
        if mode in ("full", "history"):
            result = match_scenarios(df, start_date)
            if result:
                estimates = estimate_probabilities(result["scores"], ts_code,
                                                   features=result["features"])
                matches = find_historical_match(df, estimates,
                                                exclude_after=start_date)
                print()
                print_historical_report(ts_code, df, estimates, matches)

        print(f"\n📁 报告已保存: {report_path}")
    finally:
        sys.stdout = original_stdout
        tee.close()


if __name__ == "__main__":
    main()
