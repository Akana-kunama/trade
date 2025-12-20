import sys
import os
from datetime import datetime
from pathlib import Path

# ================= 1. 路径设置 (Standard) =================
# 获取项目根目录 D:/work/trade
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# ================= 2. 导入策略 =================
from strategy_pool.selectors.policy.n_pattern_policy import NPatternSelector

def main():
    print("=" * 50)
    print(f"📅 每日选股任务启动 - {datetime.now()}")
    print("=" * 50)

    # 1. 确定选股日期
    # 如果是在盘后跑，就是今天
    # 也可以手动指定日期测试: target_date = "2025-01-10"
    target_date = datetime.now().strftime('%Y-%m-%d')
    
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