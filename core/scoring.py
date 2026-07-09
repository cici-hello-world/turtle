"""海龟策略评分模块

提供数据驱动的参数评分体系：
  - 熵权法：让数据决定各指标的权重
  - 百分位归一化：抗极端值的跨组评分（相对评分，依赖同组样本）
  - 绝对评分器：基于预校准分位数阈值的固定尺度评分（不依赖同组样本）
  - 奖惩机制：基于"超出/低于第二名比率"的边际奖惩

用法：
    from core.scoring import (entropy_weights, compute_percentile_scores,
                              apply_reward_penalty, AbsoluteScorer, create_absolute_scorer)
"""

import json
import os
import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════
# 熵权法
# ═══════════════════════════════════════════════════════════════

def entropy_weights(metrics_df: pd.DataFrame) -> dict:
    """熵权法计算各指标权重。

    信息熵越低（数据离散度越高）→ 权重越大。
    因为离散度高的指标更能区分参数优劣。

    Args:
        metrics_df: DataFrame，必须包含列 excess, robust_sharpe, r_cubed

    Returns:
        {"excess": w1, "robust_sharpe": w2, "r_cubed": w3}，权重之和 = 1.0
    """
    cols = ["excess", "robust_sharpe", "r_cubed"]
    data = metrics_df[cols].values.astype(float)

    n_samples, n_indicators = data.shape

    if n_samples < 3:
        # 样本不足时退化为默认权重
        return {"excess": 0.50, "robust_sharpe": 0.40, "r_cubed": 0.10}

    # Step 1: 每列 min-max 归一化到 [0.01, 0.99]（避免 ln(0)）
    p = np.zeros_like(data)
    for j in range(n_indicators):
        col_min = data[:, j].min()
        col_max = data[:, j].max()
        if col_max - col_min < 1e-12:
            p[:, j] = 0.5  # 所有值相等时无信息
        else:
            p[:, j] = (data[:, j] - col_min) / (col_max - col_min) * 0.98 + 0.01

    # Step 2: 计算比例 p_ij = x_ij / Σx_i
    col_sums = p.sum(axis=0)
    p_normalized = p / col_sums[np.newaxis, :]

    # Step 3: 计算熵 e_j = -k * Σ(p_ij * ln(p_ij))
    k = 1.0 / np.log(n_samples)
    entropy = np.zeros(n_indicators)
    for j in range(n_indicators):
        e_j = -k * np.sum(p_normalized[:, j] * np.log(p_normalized[:, j]))
        entropy[j] = e_j

    # Step 4: 权重 w_j = (1-e_j) / Σ(1-e_j)
    d = 1.0 - entropy  # 差异系数
    weights = d / d.sum()

    return {
        "excess": round(float(weights[0]), 4),
        "robust_sharpe": round(float(weights[1]), 4),
        "r_cubed": round(float(weights[2]), 4),
    }


# ═══════════════════════════════════════════════════════════════
# 百分位归一化（从 turtle_optimize.py 提取，参数化 weights）
# ═══════════════════════════════════════════════════════════════

def compute_percentile_scores(results: list[dict],
                               weights: dict | None = None) -> list[dict]:
    """批量百分位归一化评分。

    对每个指标在当前结果集内做 rank(pct=True)*100，
    然后按权重加权求和得到 score。

    每个结果会原地增加字段：
      - pct_excess, pct_robust_sharpe, pct_r_cubed（各指标百分位）
      - score（加权百分位总分）

    Args:
        results: 回测统计列表，每个 dict 需含 excess, robust_sharpe, r_cubed
        weights: 指标权重，默认 {"excess": 0.50, "robust_sharpe": 0.40, "r_cubed": 0.10}

    Returns:
        results（原地修改并返回）
    """
    if weights is None:
        weights = {"excess": 0.50, "robust_sharpe": 0.40, "r_cubed": 0.10}

    if len(results) < 3:
        for r in results:
            r["score"] = 50.0
        return results

    df = pd.DataFrame(results)

    for metric, w in weights.items():
        pcts = df[metric].rank(pct=True) * 100
        for i, r in enumerate(results):
            r[f"pct_{metric}"] = round(float(pcts.iloc[i]), 1)

    for r in results:
        score = sum(r.get(f"pct_{m}", 50) * w for m, w in weights.items())
        r["score"] = round(score, 2)

    return results


# ═══════════════════════════════════════════════════════════════
# 奖惩机制 — 基于"超出/低于第二名比率"
# ═══════════════════════════════════════════════════════════════

def apply_reward_penalty(results: list[dict]) -> list[dict]:
    """基于"超出/低于第二名比率"的边际奖惩。

    在每个指标上：
      - 排名第1：bonus = (value_1st - value_2nd) / value_2nd × 10，上限 +10
      - 排名最后：penalty = (value_2nd_last - value_last) / value_2nd_last × 10，上限 -10
      - 其他位置：不奖惩

    最终 score = 原百分位得分 + bonus - penalty

    每个结果会原地增加字段：
      - score（调整后）
      - bonus（奖励分）
      - penalty（惩罚分）

    Args:
        results: 已含 score 字段的结果列表

    Returns:
        results（原地修改并返回）
    """
    if len(results) < 3:
        return results

    metrics = ["excess", "robust_sharpe", "r_cubed"]

    for metric in metrics:
        values = [r[metric] for r in results]
        sorted_indices = sorted(range(len(values)), key=lambda i: values[i])

        # 排名第1：奖励
        first_idx = sorted_indices[-1]
        second_idx = sorted_indices[-2]
        val_1st = values[first_idx]
        val_2nd = values[second_idx]

        if abs(val_2nd) > 1e-10:
            bonus = min((val_1st - val_2nd) / abs(val_2nd) * 10, 10.0)
            bonus = max(bonus, 0)  # 不惩罚第1名
        else:
            bonus = 0.0

        results[first_idx].setdefault("bonus", 0.0)
        results[first_idx]["bonus"] += round(bonus, 2)

        # 排名最后：惩罚
        last_idx = sorted_indices[0]
        second_last_idx = sorted_indices[1]
        val_last = values[last_idx]
        val_2nd_last = values[second_last_idx]

        if abs(val_2nd_last) > 1e-10:
            penalty = min((val_2nd_last - val_last) / abs(val_2nd_last) * 10, 10.0)
            penalty = max(penalty, 0)
        else:
            penalty = 0.0

        results[last_idx].setdefault("penalty", 0.0)
        results[last_idx]["penalty"] += round(penalty, 2)

    # 将奖惩应用到 score
    for r in results:
        bonus = r.pop("bonus", 0.0)
        penalty = r.pop("penalty", 0.0)
        r["score"] = round(r["score"] + bonus - penalty, 2)

    return results


# ═══════════════════════════════════════════════════════════════
# 评分管线 — 一站式调用
# ═══════════════════════════════════════════════════════════════

def score_pipeline(all_results: list[dict],
                   fixed_weights: dict | None = None) -> tuple[list[dict], dict]:
    """完整评分管线：key映射 → 剔除0交易 → 熵权 → 百分位 → 奖惩。

    Args:
        all_results: 回测统计列表，每个 dict 需含 _window, _params,
                     超额收益 (%), 稳健夏普比率, R立方, 总交易
        fixed_weights: 若提供，跳过熵权计算，直接使用此权重
                       (e.g. 来自 turtle_scoring.py 的固定权重)

    Returns:
        (scored_results, weights) — results 原地修改，weights 是实际使用的权重
    """
    # Step 1: 剔除总交易=0
    before = len(all_results)
    all_results = [r for r in all_results if r.get("总交易", 0) > 0]
    if len(all_results) < 3:
        for r in all_results:
            r["score"] = 50.0
        fallback = {"excess": 0.50, "robust_sharpe": 0.40, "r_cubed": 0.10}
        return all_results, fixed_weights if fixed_weights else fallback

    # Step 2: 映射指标 key（中文 → 英文）
    for r in all_results:
        r.setdefault("excess", r.get("超额收益 (%)", 0.0) or 0.0)
        r.setdefault("robust_sharpe", r.get("稳健夏普比率", 0.0) or 0.0)
        r.setdefault("r_cubed", r.get("R立方", 0.0) or 0.0)

    # Step 3: 熵权法（若提供了 fixed_weights 则跳过）
    if fixed_weights:
        weights = fixed_weights
    else:
        full_window_results = [r for r in all_results if r["_window"] == "full"]
        if len(full_window_results) >= 3:
            metrics_df = pd.DataFrame([
                {"excess": r["超额收益 (%)"], "robust_sharpe": r["稳健夏普比率"],
                 "r_cubed": r["R立方"]}
                for r in full_window_results
            ])
            weights = entropy_weights(metrics_df)
        else:
            weights = {"excess": 0.50, "robust_sharpe": 0.40, "r_cubed": 0.10}

    # Step 4: 百分位归一化（所有窗口拉通）
    compute_percentile_scores(all_results, weights)

    # Step 5: 奖惩
    apply_reward_penalty(all_results)

    return all_results, weights


# ═══════════════════════════════════════════════════════════════
# 绝对评分器 — 基于预校准分位数阈值，评分不受同组样本影响
# ═══════════════════════════════════════════════════════════════

class AbsoluteScorer:
    """绝对评分器 —— 基于预校准的分位数阈值给出固定尺度评分。

    与 compute_percentile_scores（组内排名）不同，AbsoluteScorer 对每个
    参数组合给出与同批次其他组合无关的绝对分数。这消除了 SWEEP_RANGES
    配置对评分的影响 —— 同一个参数组合，不管放在哪个 sweep 集合里，
    得分始终不变。

    评分方法：对每个指标，将原始值线性映射到 [1, 99] 区间，
    锚点为校准样本的 P5 → 5分、P95 → 95分，超出范围截断。
    最终得分 = Σ(指标得分 × 熵权)。

    使用方式：
        # 从校准文件创建
        scorer = create_absolute_scorer("result/calibration_09992_HK.json")
        score = scorer.score_single(stats)         # 单条评分
        results = scorer.score_batch(results)       # 批量评分
    """

    def __init__(self, calibration: dict):
        """
        Args:
            calibration: calibrate() 或 load_calibration() 返回的 dict，
                         必须包含 weights 和 scoring_params。
        """
        self.weights = calibration.get("weights", {
            "excess": 0.50, "robust_sharpe": 0.40, "r_cubed": 0.10})
        self.scoring_params = calibration.get("scoring_params", {})
        self.quantiles = calibration.get("quantiles", {})
        self.metadata = calibration.get("metadata", {})

    def score_single(self, stats: dict) -> float:
        """对单个回测统计 dict 计算绝对评分 (0-100)。

        不依赖任何外部排名，同一 stats 永远返回相同分数。

        Args:
            stats: 回测统计 dict，需含 excess / robust_sharpe / r_cubed
                   或其中文别名（超额收益 (%) / 稳健夏普比率 / R立方）

        Returns:
            0-100 的绝对评分
        """
        # 兼容中文 key
        excess = stats.get("excess", stats.get("超额收益 (%)", 0.0) or 0.0)
        robust_sharpe = stats.get("robust_sharpe",
                                   stats.get("稳健夏普比率", 0.0) or 0.0)
        r_cubed = stats.get("r_cubed", stats.get("R立方", 0.0) or 0.0)

        raw_values = {
            "excess": float(excess),
            "robust_sharpe": float(robust_sharpe),
            "r_cubed": float(r_cubed),
        }

        score = 0.0
        for metric in ["excess", "robust_sharpe", "r_cubed"]:
            val = raw_values[metric]
            sp = self.scoring_params.get(metric, {})

            if sp.get("type") == "linear_quantile":
                slope = sp["slope"]
                intercept = sp["intercept"]
                metric_score = slope * val + intercept
                # 截断到 [1, 99]
                metric_score = max(1.0, min(99.0, metric_score))
            else:
                # constant 或其他未知类型
                metric_score = sp.get("value", 50.0)

            score += metric_score * self.weights.get(metric, 0.333)

        return round(score, 2)

    def score_batch(self, results: list[dict]) -> list[dict]:
        """批量绝对评分，每个结果原地增加 'score' 字段。

        Args:
            results: 回测统计列表

        Returns:
            results（原地修改并返回）
        """
        for r in results:
            r["score"] = self.score_single(r)
        return results

    def to_dict(self) -> dict:
        """导出为可 JSON 序列化的 dict。"""
        return {
            "weights": self.weights,
            "scoring_params": self.scoring_params,
            "quantiles": self.quantiles,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        w = self.weights
        return (f"AbsoluteScorer(excess={w['excess']:.3f}, "
                f"robust_sharpe={w['robust_sharpe']:.3f}, "
                f"r_cubed={w['r_cubed']:.3f})")


def create_absolute_scorer(calibration_source: str | dict) -> AbsoluteScorer:
    """工厂函数：从校准文件路径或校准 dict 创建 AbsoluteScorer。

    Args:
        calibration_source: JSON 文件路径 (str) 或校准 dict

    Returns:
        AbsoluteScorer 实例

    Examples:
        # 从文件加载
        scorer = create_absolute_scorer("result/calibration_09992_HK.json")

        # 直接传入校准 dict
        from core.calibrate import calibrate
        cal = calibrate(base_kwargs, df, bm)
        scorer = create_absolute_scorer(cal)
    """
    if isinstance(calibration_source, str):
        if not os.path.exists(calibration_source):
            raise FileNotFoundError(f"校准文件不存在: {calibration_source}")
        with open(calibration_source, "r", encoding="utf-8") as f:
            calibration = json.load(f)
    else:
        calibration = calibration_source

    return AbsoluteScorer(calibration)


# ═══════════════════════════════════════════════════════════════
# 带绝对评分选项的评分管线
# ═══════════════════════════════════════════════════════════════

def score_pipeline_v2(all_results: list[dict],
                       fixed_weights: dict | None = None,
                       absolute_scorer: AbsoluteScorer | None = None,
                       use_penalty: bool = True) -> tuple[list[dict], dict]:
    """增强版评分管线：支持绝对评分器 + 可选奖惩。

    与 score_pipeline 的区别：
      - 支持传入 AbsoluteScorer，此时跳过熵权和百分位排名，
        直接使用绝对评分
      - 奖惩变为可选（绝对评分下奖惩意义不同）

    Args:
        all_results: 回测统计列表
        fixed_weights: 固定权重（用于百分位模式）
        absolute_scorer: 绝对评分器（优先使用）
        use_penalty: 是否应用奖惩机制

    Returns:
        (scored_results, weights)
    """
    # Step 1: 剔除总交易=0
    all_results = [r for r in all_results if r.get("总交易", 0) > 0]
    if len(all_results) < 3:
        for r in all_results:
            r["score"] = 50.0
        fallback = {"excess": 0.50, "robust_sharpe": 0.40, "r_cubed": 0.10}
        return all_results, fallback

    # Step 2: 映射指标 key
    for r in all_results:
        r.setdefault("excess", r.get("超额收益 (%)", 0.0) or 0.0)
        r.setdefault("robust_sharpe", r.get("稳健夏普比率", 0.0) or 0.0)
        r.setdefault("r_cubed", r.get("R立方", 0.0) or 0.0)

    # Step 3: 评分
    if absolute_scorer is not None:
        # 绝对评分模式：不依赖同组样本
        absolute_scorer.score_batch(all_results)
        weights = absolute_scorer.weights
    elif fixed_weights:
        # 百分位模式 + 固定权重
        weights = fixed_weights
        compute_percentile_scores(all_results, weights)
    else:
        # 百分位模式 + 熵权
        full_window_results = [r for r in all_results if r["_window"] == "full"]
        if len(full_window_results) >= 3:
            metrics_df = pd.DataFrame([
                {"excess": r.get("excess", 0), "robust_sharpe": r.get("robust_sharpe", 0),
                 "r_cubed": r.get("r_cubed", 0)}
                for r in full_window_results
            ])
            weights = entropy_weights(metrics_df)
        else:
            weights = {"excess": 0.50, "robust_sharpe": 0.40, "r_cubed": 0.10}
        compute_percentile_scores(all_results, weights)

    # Step 4: 奖惩（可选）
    if use_penalty:
        apply_reward_penalty(all_results)

    return all_results, weights
