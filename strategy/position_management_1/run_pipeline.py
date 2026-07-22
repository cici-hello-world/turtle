#!/usr/bin/env python3
"""一键依次执行：calibrate → turtle_scoring → turtle_sweep

用法：
  python strategy/position_management_1/run_pipeline.py                     # 遍历所有标的
  python strategy/position_management_1/run_pipeline.py --ts-code 09992.HK  # 单只标的
  python strategy/position_management_1/run_pipeline.py --n-samples 200 --skip-calibrate
"""

import argparse
import subprocess
import sys
import os
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from strategy.position_management_1.common_config import STOCKS


def run_step(name: str, cmd: list[str]) -> bool:
    """运行一个步骤，返回是否成功。"""
    print(f"\n{'=' * 70}")
    print(f"🚀 {name}")
    print(f"{'=' * 70}")
    print(f"  命令: {' '.join(cmd)}")
    print()

    t0 = time.time()
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    elapsed = time.time() - t0

    if result.returncode == 0:
        print(f"\n✅ {name} 完成 ({elapsed / 60:.1f}min)")
        return True
    else:
        print(f"\n❌ {name} 失败 (exit code={result.returncode}, {elapsed / 60:.1f}min)")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="一键执行 calibrate → turtle_scoring → turtle_sweep")
    parser.add_argument("--ts-code", default=None,
                        help="股票代码（不传则遍历所有标的）")
    parser.add_argument("--n-samples", type=int, default=500,
                        help="calibrate LHS 采样数 (默认: 500)")
    parser.add_argument("--workers", type=int, default=None,
                        help="并行进程数 (默认: CPU-1)")
    parser.add_argument("--skip-calibrate", action="store_true",
                        help="跳过 calibrate，直接从 scoring 开始")
    parser.add_argument("--skip-scoring", action="store_true",
                        help="跳过 turtle_scoring，直接从 sweep 开始")
    parser.add_argument("--serial", action="store_true",
                        help="scoring/sweep 使用串行模式")
    args = parser.parse_args()

    # 确定要跑的标的列表
    if args.ts_code:
        ts_codes = [args.ts_code]
    else:
        ts_codes = list(STOCKS.keys())

    pipeline_start = time.time()

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  海龟策略一键流水线                                           ║")
    print("║  calibrate → turtle_scoring → turtle_sweep                  ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"  标的数量: {len(ts_codes)}")
    print(f"  标的列表: {', '.join(ts_codes)}")
    print(f"  calibrate 采样数: {args.n_samples}")
    print()

    failed_stocks = []

    for idx, ts_code in enumerate(ts_codes, 1):
        stock_start = time.time()
        stock_name = STOCKS.get(ts_code, {}).get("stock_name", ts_code)

        print(f"\n{'#' * 70}")
        print(f"##  [{idx}/{len(ts_codes)}] {stock_name} ({ts_code})")
        print(f"{'#' * 70}")

        # ── Step 1: calibrate ──
        if not args.skip_calibrate:
            cmd = [
                sys.executable,
                os.path.join(PROJECT_ROOT, "core", "calibrate.py"),
                "--ts-code", ts_code,
                "--n-samples", str(args.n_samples),
                "--seed", "42",
            ]
            if args.workers:
                cmd += ["--workers", str(args.workers)]
            if not run_step(f"Step 1/3: calibrate (全域采样 + 熵权计算)", cmd):
                print(f"\n⚠️ {ts_code} calibrate 失败，跳过该标的剩余步骤")
                failed_stocks.append(ts_code)
                continue
        else:
            print(f"\n⏭️  跳过 Step 1/3: calibrate")

        # ── Step 2: turtle_scoring ──
        if not args.skip_scoring:
            cmd = [
                sys.executable,
                os.path.join(PROJECT_ROOT, "strategy", "position_management_1",
                             "turtle_scoring.py"),
                "--ts-code", ts_code,
            ]
            if args.workers:
                cmd += ["--workers", str(args.workers)]
            if args.serial:
                cmd.append("--serial")
            if not run_step(f"Step 2/3: turtle_scoring (三窗口统一评分)", cmd):
                print(f"\n⚠️ {ts_code} turtle_scoring 失败，跳过该标的剩余步骤")
                failed_stocks.append(ts_code)
                continue
        else:
            print(f"\n⏭️  跳过 Step 2/3: turtle_scoring")

        # ── Step 3: turtle_sweep ──
        cmd = [
            sys.executable,
            os.path.join(PROJECT_ROOT, "strategy", "position_management_1",
                         "turtle_sweep.py"),
            "--ts-code", ts_code,
        ]
        if args.workers:
            cmd += ["--workers", str(args.workers)]
        if args.serial:
            cmd.append("--serial")
        if not run_step(f"Step 3/3: turtle_sweep (单参数精细扫描)", cmd):
            print(f"\n⚠️ {ts_code} turtle_sweep 失败，跳过该标的剩余步骤")
            failed_stocks.append(ts_code)
            continue

        stock_elapsed = (time.time() - stock_start) / 60
        print(f"\n✅ {stock_name} ({ts_code}) 完成 ({stock_elapsed:.1f}min)")

    # ── 汇总 ──
    total_elapsed = (time.time() - pipeline_start) / 60
    succeeded = [c for c in ts_codes if c not in failed_stocks]

    print(f"\n{'=' * 70}")
    print(f"🎉 流水线全部完成！")
    print(f"  成功: {len(succeeded)}/{len(ts_codes)}  →  {', '.join(succeeded) if succeeded else '无'}")
    if failed_stocks:
        print(f"  失败: {len(failed_stocks)}/{len(ts_codes)}  →  {', '.join(failed_stocks)}")
    print(f"  总耗时: {total_elapsed:.1f}min")
    print(f"{'=' * 70}")
    print(f"  共享配置: strategy/position_management_1/config_state.json")


if __name__ == "__main__":
    main()
