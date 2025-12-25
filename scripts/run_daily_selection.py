import sys
import os
from datetime import datetime
from pathlib import Path
from datetime import datetime
import argparse
# ================= 1. 路径设置 (Standard) =================
# 获取项目根目录 D:/work/trade
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# ================= 2. 导入策略 =================
from strategy_pool.selectors.policy.n_pattern_policy import NPatternSelector

def parse_args():
    parser = argparse.ArgumentParser(description="每日选股任务")
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="选股日期，格式 YYYY-MM-DD；不传则默认为今天"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 50)
    print(f"📅 每日选股任务启动 - {datetime.now()}")
    print("=" * 50)

    # 1. 确定选股日期
    if args.date:
        target_date = args.date
        print(f"📌 使用指定选股日期: {target_date}")
    else:
        target_date = datetime.now().strftime('%Y-%m-%d')
        print(f"📌 使用当前日期: {target_date}")

    # 2. 初始化策略
    try:
        strategy = NPatternSelector()
    except Exception as e:
        print(f"❌ 策略初始化失败: {e}")
        return

    # 3. 运行策略
    strategy.run(date=target_date)

    print("\n✅ 选股任务结束")


if __name__ == "__main__":
    main()