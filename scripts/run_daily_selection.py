# scripts/run_daily_selection.py
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from strategy_pool.selectors.policy.n_pattern_policy import NPatternSelector

def main():
    print(">>> 开始执行每日选股任务...")

    # ====== 1. 解析命令行日期参数 ======
    # 用法：
    #   python n_pattern_policy.py              → 使用最新行情
    #   python n_pattern_policy.py 2024-01-05   → 指定日期
    target_date = None

    if len(sys.argv) >= 2:
        arg = sys.argv[1]
        try:
            target_date = datetime.strptime(arg, "%Y-%m-%d")
            print(f"📅 使用指定选股日: {target_date.date()}")
        except ValueError:
            print(f"⚠️ 日期格式错误，应为 YYYY-MM-DD，例如：2024-01-05")
            return

    # ====== 2. 执行策略 ======
    try:
        strategy = NPatternSelector()
        strategy.run(date=target_date)
    except Exception as e:
        print(f"❌ NPatternSelector 运行失败: {e}")

    print(">>> 选股任务结束")


if __name__ == "__main__":
    main()