#!/usr/bin/env python3
"""每日仓位信号 — crontab 18:00 调用

用法:
    python scripts/daily_signal.py           # 默认：更新缓存 + 提取信号 + 推送
    python scripts/daily_signal.py --dry-run  # 仅打印，不推送
"""

import sys
import os

# 自动加载项目根目录 .env（crontab 环境不走 shell profile）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ENV_FILE = os.path.join(PROJECT_ROOT, ".env")
if os.path.isfile(_ENV_FILE):
    with open(_ENV_FILE) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _val = _line.split("=", 1)
                _val = _val.strip().strip('"').strip("'")
                os.environ.setdefault(_key.strip(), _val)

sys.path.insert(0, PROJECT_ROOT)

from datetime import datetime

from core.config import BacktestConfig
from core.notify import send as notify_send
from core.signal_extractor import extract_signal, format_signal

# 导入缓存更新
from data_cache.update_cache import main as update_cache_main


# ── 股票配置 ──────────────────────────────────────────────
STOCK_CONFIGS = [
    BacktestConfig(
        ts_code="09992.HK",
        stock_name="泡泡玛特",
        market="HK",
        mode="position_management_1",
        per_shares=200,
        pm_start_date="20260629",
        start_date="20201211",
        user_dmi=True,
        pm_dynamic_position=False,
        pm_adaptive_grid=False,
        ema_fast=0,
        ema_slow=0,
        pm_grid_down=0.15, pm_grid_up=0.065, pm_reduce_pct=0.67,
        pm_add_pct=0.42, pm_adx_threshold=37, pm_agg_k=1,
        pm_initial_shares=2,
        initial_capital=40000,
        data_source="tushare",
    ),
    BacktestConfig(
        ts_code="02015.HK",
        stock_name="理想",
        market="HK",
        mode="position_management_1",
        per_shares=100,
        pm_start_date="20260629",
        start_date="20220813",
        user_dmi=True,
        pm_dynamic_position=False,
        pm_adaptive_grid=False,
        ema_fast=0,
        ema_slow=0,
        pm_grid_down=0.05, pm_grid_up=0.04, pm_reduce_pct=0.75,
        pm_add_pct=0.50, pm_adx_threshold=24, pm_agg_k=1,
        pm_initial_shares=25,
        initial_capital=40000,
        data_source="tushare",
    ),
    BacktestConfig(
        ts_code="03690.HK",
        stock_name="美团",
        market="HK",
        mode="position_management_1",
        per_shares=100,
        pm_start_date="20260629",
        start_date="20180920",
        user_dmi=True,
        pm_adaptive_grid = False,
        pm_dynamic_position=False,
        ema_fast=0,
        ema_slow=0,
        pm_grid_down=0.10, pm_grid_up=0.039, pm_reduce_pct=0.51,
        pm_add_pct=0.5, pm_adx_threshold=35, pm_agg_k=1,
        pm_initial_shares=2,
        initial_capital=40000,
        data_source="tushare",
    ),
    BacktestConfig(
        ts_code="01810.HK",
        stock_name="小米",
        market="HK",
        mode="position_management_1",
        per_shares=100,
        pm_start_date="20260629",
        start_date="20180709",
        user_dmi=True,
        pm_adaptive_grid=False,
        pm_dynamic_position=False,
        ema_fast=0,
        ema_slow=0,
        pm_grid_down=0.07, pm_grid_up=0.05, pm_reduce_pct=0.70,
        pm_add_pct=0.36, pm_adx_threshold=27, pm_agg_k=1,
        pm_initial_shares=8,
        initial_capital=40000,
        data_source="tushare",
    ),
    BacktestConfig(
        ts_code="06181.HK",
        stock_name="老铺",
        market="HK",
        mode="position_management_1",
        per_shares=100,
        pm_start_date="20260629",
        start_date="20240628",
        user_dmi=True,
        pm_adaptive_grid=True,
        ema_fast=0,
        ema_slow=0,
        pm_grid_down=0.1, pm_grid_up=0.028, pm_reduce_pct=0.75,
        pm_add_pct=0.50, pm_adx_threshold=30, pm_agg_k=0.8,
        pm_initial_shares=2,
        initial_capital=40000,
        data_source="tushare",
    ),
]


def main():
    dry_run = "--dry-run" in sys.argv

    today = datetime.today()
    print(f"🐢 海龟每日信号 — {today.strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    # ── Step 1: 更新缓存 ──
    print("\n📡 Step 1: 更新缓存\n")
    # 隔离 sys.argv，防止 daily_signal 的参数干扰 update_cache 解析
    old_argv = sys.argv
    sys.argv = ["update_cache"]
    cache_results = update_cache_main()
    sys.argv = old_argv

    if not cache_results:
        print("⚠️ 无股票需要更新")
        return

    failed = [k for k, v in cache_results.items() if not v]
    if failed:
        print(f"\n⚠️ {len(failed)} 只股票缓存更新失败: {', '.join(failed)}")

    # ── Step 2: 提取信号 ──
    print(f"\n📊 Step 2: 提取仓位信号\n")
    signals = []
    for config in STOCK_CONFIGS:
        # 跳过缓存失败的股票
        if config.ts_code in failed:
            print(f"  ⏭️ {config.stock_name} ({config.ts_code}) 缓存失败，跳过")
            continue

        signal = extract_signal(config)
        if signal:
            signals.append(signal)
        else:
            print(f"  ❌ {config.stock_name} ({config.ts_code}) 信号提取失败")

    if not signals:
        print("\n⚠️ 无有效信号")
        return

    # ── Step 3: 组装通知消息 ──
    print(f"\n📨 Step 3: 推送通知\n")

    signal_blocks = [format_signal(s) for s in signals]

    body_parts = [f"## 🐢 海龟信号 {today.strftime('%Y-%m-%d')}"]
    body_parts.append("---")
    body_parts.extend(signal_blocks)

    if failed:
        body_parts.append("---")
        body_parts.append(f"### ⚠️ 缓存失败")
        body_parts.extend(f"  • {ts_code}" for ts_code in failed)

    body = "\n\n".join(body_parts)

    if dry_run:
        print("── 消息预览（--dry-run）──")
        print(body)
        print("── 结束 ──")
    else:
        ok = notify_send(f"🐢 海龟信号 {today.strftime('%m/%d')}", body)
        if ok:
            print("✅ 通知已发送")
        else:
            print("❌ 通知发送失败")


if __name__ == "__main__":
    main()
