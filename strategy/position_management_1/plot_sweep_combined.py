#!/usr/bin/env python3
"""将单参数扫描的5个Excel结果合并为一张多子图"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti SC', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

RESULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "result")

PARAM_FILES = {
    "pm_grid_down":      ("网格向下间距 gd", "turtle_sweep_pm_grid_down.xlsx"),
    "pm_grid_up":        ("网格向上间距 gu", "turtle_sweep_pm_grid_up.xlsx"),
    "pm_reduce_pct":     ("减仓比例 rp",      "turtle_sweep_pm_reduce_pct.xlsx"),
    "pm_add_pct":        ("加仓比例 ap",      "turtle_sweep_pm_add_pct.xlsx"),
    "pm_adx_threshold":  ("ADX阈值 adx",      "turtle_sweep_pm_adx_threshold.xlsx"),
}

METRICS = [
    ("score",          "综合得分",       "steelblue",   "-", "o"),
    ("r_cubed",        "R³",            "darkorange",  "--", "s"),
    ("robust_sharpe",  "稳健夏普",       "seagreen",    "-.", "D"),
    ("excess",         "超额收益%",      "firebrick",   ":",  "^"),
]

def _minmax_norm(raw):
    lo, hi = min(raw), max(raw)
    if hi - lo < 1e-12:
        return [50.0] * len(raw)
    return [(v - lo) / (hi - lo) * 100 for v in raw]

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, (pname, (label, filename)) in enumerate(PARAM_FILES.items()):
    ax = axes[idx]
    filepath = os.path.join(RESULT_DIR, filename)
    if not os.path.exists(filepath):
        ax.set_title(f"{label} — 数据缺失")
        continue

    df = pd.read_excel(filepath)
    df = df.sort_values("value")

    x = df["value"].values
    for key, disp, color, ls, marker in METRICS:
        if key not in df.columns:
            continue
        raw = df[key].values
        y = _minmax_norm(raw)
        ax.plot(x, y, color=color, linestyle=ls, marker=marker,
                linewidth=1.5, markersize=4, label=disp)

    # 标注最高点
    best = df.loc[df["score"].idxmax()]
    scores_norm = _minmax_norm(df["score"].values)
    best_y = scores_norm[df["score"].values.argmax()]
    ax.annotate(f'{best["value"]:.3f}',
                xy=(best["value"], best_y),
                xytext=(0, -12), textcoords="offset points",
                fontsize=9, color="darkred", ha="center",
                arrowprops=dict(arrowstyle="->", color="darkred", lw=0.8))

    ax.set_xlabel(label, fontsize=12)
    ax.set_ylabel("归一化得分 (0–100)", fontsize=11)
    ax.set_title(label, fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

# 隐藏多余子图
for j in range(len(PARAM_FILES), len(axes)):
    axes[j].set_visible(False)

fig.suptitle("09992.HK 泡泡玛特 — 单参数扫描 (三类窗口综合评分)",
             fontsize=15, fontweight="bold", y=1.01)
fig.tight_layout()

out_path = os.path.join(RESULT_DIR, "turtle_sweep_all.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"✅ 合并图表已保存: {out_path}")
