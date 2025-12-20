# config/path_config.py
# -*- coding: utf-8 -*-
from pathlib import Path
import os

# ================= 项目根目录定位 =================
# 获取当前文件 (path_config.py) 的父目录 (config) 的父目录 (trade2)
# 这样无论你在哪里运行，路径都是对的
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# ================= 数据中心路径 (Data Center) =================
DATA_ROOT = PROJECT_ROOT / "data_center" / "storage"

# 1. 动态行情 (Parquet/DB 文件存放处)
# 原来的: data/financial_data -> 现在建议分得更细
MARKET_DATA_DIR = DATA_ROOT / "market_data"
MARKET_DATA_DIR.mkdir(parents=True, exist_ok=True)

# 2. 静态基础信息 (股票列表、行业分类)
# 原来的: data/stockinfo_map
BASIC_INFO_DIR = DATA_ROOT / "basic_info"
BASIC_INFO_DIR.mkdir(parents=True, exist_ok=True)

# ================= 策略与产出路径 =================
# 策略产出根目录
STRATEGY_WORKSPACE = PROJECT_ROOT / "strategy_pool" / "selectors" / "pool_storage"
STRATEGY_WORKSPACE.mkdir(parents=True, exist_ok=True)

# 临时文件夹 (存放缓存)
TEMP_DIR = PROJECT_ROOT / "temp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# 日志目录
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ... (保留原有的 PROJECT_ROOT, DATA_ROOT 等定义)

# ================= 因子工厂路径 (Factor Lab) =================
# 因子库根目录
FACTOR_LAB_DIR = PROJECT_ROOT / "factor_lab"
# 因子存储目录
FACTOR_STORE_DIR = FACTOR_LAB_DIR / "storage" / "stock_factors"
# 市场因子文件
MARKET_FACTOR_PATH = FACTOR_LAB_DIR / "storage" / "market_factors.csv"

# ================= 模型工厂路径 (Model Factory) =================
MODEL_FACTORY_DIR = PROJECT_ROOT / "model_factory"
# 模型仓库 (Model Zoo)
MODEL_ZOO_DIR = MODEL_FACTORY_DIR / "model_zoo"
MODEL_ZOO_DIR.mkdir(parents=True, exist_ok=True)

print(f"✅ 项目根目录定位: {PROJECT_ROOT}")