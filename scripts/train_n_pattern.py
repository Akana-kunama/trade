import sys
import pandas as pd
import numpy as np
import re
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# 路径 Hack
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from model_factory.data_loader import DatasetLoader
from model_factory.registry import ModelRegistry
from model_factory.models.lgbm_algo import LGBMClassifierModel
from model_factory.models.xgboost_algo import XGBoostModel
from model_factory.models.catboost_algo import CatBoostModel

# ================= 🔧 训练配置 =================
# 1. 实验名称 (使用 "/" 来创建子文件夹分类)
# scripts/train_n_pattern.py
# ... (Imports 保持不变) ...

# ================= 🔧 训练配置 =================
EXPERIMENT_GROUP = "n_pattern"
VERSION = "v1"

FEATURES = [
    'ma5_bias', 'vol_ratio', 'upper_shadow', 'body_size', 
    'high_low_ratio', 'market_limit_up_count' 
]

# 定义我们要训练的两个目标
TARGET_LABELS = ["label_strategy", "label_short_term"]

BASE_CONFIG = {
    "start_date": '2020-01-01',
    "end_date": '2025-09-18',
    "filter": { "column": "n_pattern_signal", "value": 1 },
    "features": FEATURES,
    "codes": None 
}

VALID_START_DATE = '2025-01-01'

def run_training():
    print(f"🚀 启动 N字反转多目标训练任务...")
    
    # 1. 加载数据 (一次性加载所有需要的列，避免重复IO)
    loader = DatasetLoader()
    from config.path_config import FACTOR_STORE_DIR
    all_codes = [f.stem for f in FACTOR_STORE_DIR.glob("*.parquet")]
    BASE_CONFIG['codes'] = all_codes
    
    # 修改 config 让它去请求所有 label
    load_config = BASE_CONFIG.copy()
    # 这里的 label 只是个占位，为了让 loader 去取数，我们手动把 TARGET_LABELS 加入 features 列表传进去
    # 这是一个小 trick，因为 DatasetLoader 只会取 config['features'] + config['label']
    load_config['features'] = FEATURES + TARGET_LABELS 
    load_config['label'] = TARGET_LABELS[0] # 随便给一个
    
    df_all = loader.load_dataset(load_config)
    if df_all.empty: return

    # 2. 切分
    df_train = df_all[df_all['date'] < VALID_START_DATE]
    df_valid = df_all[df_all['date'] >= VALID_START_DATE]
    print(f"📊 训练集: {len(df_train)} | 验证集: {len(df_valid)}")

    # 3. 循环训练不同 Label 的模型
    final_ids = {}
    
    for label_name in TARGET_LABELS:
        print(f"\n🎯 正在训练目标: [{label_name}]")
        
        # 准备数据
        X_train = df_train[FEATURES]
        y_train = df_train[label_name]  # 动态取 Label
        X_valid = df_valid[FEATURES]
        y_valid = df_valid[label_name]
        
        # 训练该 Label 下的三大模型
        registry = ModelRegistry()
        models = {
            "lgbm": LGBMClassifierModel(),
            "xgb": XGBoostModel(),
            "cat": CatBoostModel()
        }
        
        for algo_name, model in models.items():
            print(f"   🤖 Training {algo_name} ...")
            model.fit(X_train, y_train, X_valid, y_valid)
            
            # 评估
            score = model.predict(X_valid)
            auc = roc_auc_score(y_valid, score)
            print(f"      AUC: {auc:.4f}")
            
            # 保存：文件夹名带上 label，如 n_pattern/v1_cat_2025..._label_strategy
            # 这里我们在 experiment_name 后缀加上 label
            exp_name = f"{EXPERIMENT_GROUP}/{VERSION}_{algo_name}_{label_name}"
            
            # 保存配置也要更新 label
            save_config = BASE_CONFIG.copy()
            save_config['label'] = label_name
            
            exp_id = registry.save_experiment(
                model, save_config, {"auc": auc}, exp_name
            )
            
            # 记录 ID
            key = f"{label_name}_{algo_name}" # key = "label_strategy_lgbm"
            final_ids[key] = exp_id

    print("\n✅ 所有模型训练完成！请更新策略配置：")
    print("=" * 60)
    for k, v in final_ids.items():
        print(f'"{k}": "{v}",')
    print("=" * 60)

if __name__ == "__main__":
    run_training()