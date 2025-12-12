
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from strategy_pool.selectors.analyzer.analyzer import SelectorAnalyzer
from strategy_pool.selectors.policy.n_pattern_policy import NPatternSelector

def main():
    analyzer = SelectorAnalyzer(
        selector_cls=NPatternSelector,
        start_date="2025-01-01",
        end_date="2025-12-01"
    )
    df_stats = analyzer.run_count_per_day(
        overwrite=False,   # 已经有的日结果就直接复用
        freq="B",          # 工作日
        plot=True          # 自动画图 & 保存
    )

if __name__ == "__main__":
    main()
