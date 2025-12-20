import sys
import os
import glob
from pathlib import Path

# ================= 路径 Hack (确保能找到项目模块) =================
# 获取脚本所在目录的上一级目录 (即项目根目录 D:\work\trade)
CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_SCRIPT_DIR.parent
sys.path.append(str(PROJECT_ROOT))

# ================= 导入模块 =================
# 从你的配置文件导入正确的变量名
from config.path_config import MARKET_DATA_DIR, PROJECT_ROOT

from factor_lab.engine.stock_updater import StockFactorUpdater
from factor_lab.engine.market_updater import MarketFactorUpdater

# ================= 路径配置 =================
# 1. 原始行情路径 (对应 stock_daily 文件夹)
# 你的 path_config 中 MARKET_DATA_DIR 指向 storage/market_data
RAW_DATA_DIR = MARKET_DATA_DIR / "stock_daily"

# 2. 因子库存储路径 (新建在 factor_lab 下)
FACTOR_LAB_DIR = PROJECT_ROOT / "factor_lab" / "storage"
FACTOR_STORE_DIR = FACTOR_LAB_DIR / "stock_factors"
MARKET_FACTOR_PATH = FACTOR_LAB_DIR / "market_factors.csv"

# 确保目录存在
if not RAW_DATA_DIR.exists():
    print(f"❌ 错误: 原始数据目录不存在: {RAW_DATA_DIR}")
    sys.exit(1)

FACTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print(f"📂 原始数据路径: {RAW_DATA_DIR}")
    print(f"📂 因子输出路径: {FACTOR_STORE_DIR}")
    print("=" * 50)

    # --- 第一步: 更新全市场因子 (如涨跌停家数) ---
    print("\n[Step 1/2] Updating Market-Level Factors...")
    market_updater = MarketFactorUpdater(str(RAW_DATA_DIR), str(MARKET_FACTOR_PATH))
    market_updater.run()

    # --- 第二步: 更新个股因子 (MA5, VolRatio, Labels) ---
    print("\n[Step 2/2] Updating Stock-Level Factors & Labels...")
    
    # 获取所有parquet文件
    all_files = list(RAW_DATA_DIR.glob("*.parquet"))
    
    if not all_files:
        print("⚠️  警告: 没有找到任何行情文件，请先运行数据更新脚本。")
        return

    # 提取股票代码 (去掉后缀)
    all_codes = [f.stem for f in all_files] # .stem 自动去掉 .parquet 后缀
    
    # === 测试模式 (可选) ===
    # 为了快速跑通，这里先只跑前 10 只股票。
    # 确认没问题后，把下面这行注释掉，跑全量。
    test_codes = all_codes[:20] 
    # print(f"🧪 测试模式: 仅处理前 {len(test_codes)} 只股票...")

    # 初始化更新器
    stock_updater = StockFactorUpdater(str(RAW_DATA_DIR), str(FACTOR_STORE_DIR))
    
    # 运行批量更新
    # 正式运行时改为: stock_updater.run_batch(stock_codes=all_codes)
    stock_updater.run_batch(stock_codes=all_codes)

    print("\n✅ Factor Pipeline Finished Successfully!")

if __name__ == "__main__":
    main()