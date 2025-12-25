import os
import json
import joblib
import datetime
import glob
from pathlib import Path
from config.path_config import MODEL_ZOO_DIR

# 引入你的模型类
from .models.lgbm_algo import LGBMClassifierModel
from .models.xgboost_algo import XGBoostModel
from .models.catboost_algo import CatBoostModel

class ModelRegistry:
    def __init__(self):
        self.root = MODEL_ZOO_DIR
        if not self.root.exists():
            self.root.mkdir(parents=True, exist_ok=True)

    def save_experiment(self, model_obj, config, metrics, experiment_name):
        '''
        保存模型。
        experiment_name 支持路径格式，如 "n_pattern/v1_lgbm"
        '''
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 处理 experiment_name 中的路径分隔符
        # 例如: n_pattern/v1_lgbm -> 文件夹名: n_pattern/v1_lgbm_20251220_xxxx
        # 为了保持文件夹结构清晰，我们把时间戳加在最后的文件名部分
        
        parts = experiment_name.split('/')
        base_name = parts[-1] # v1_lgbm
        sub_dirs = parts[:-1] # ['n_pattern']
        
        # 目标文件夹名
        final_folder_name = f"{base_name}_{timestamp}"
        
        # 拼接完整路径: model_zoo / n_pattern / v1_lgbm_timestamp
        save_dir = self.root
        for sub in sub_dirs:
            save_dir = save_dir / sub
        
        save_dir = save_dir / final_folder_name
        save_dir.mkdir(parents=True, exist_ok=True)

        # 1. 保存模型
        model_path = save_dir / "model.pkl"
        model_obj.save(str(model_path))

        # 2. 保存元数据
        metadata = {
            "experiment_id": str(save_dir.relative_to(self.root)).replace("\\", "/"), # 保存相对路径作为ID
            "timestamp": timestamp,
            "model_class": model_obj.__class__.__name__,
            "config": config,
            "metrics": metrics
        }
        
        with open(save_dir / "meta.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)
            
        print(f"💾 模型已保存: {metadata['experiment_id']}")
        return metadata['experiment_id']

    def load_model(self, experiment_id):
        '''根据 ID (相对路径) 加载模型'''
        model_dir = self.root / experiment_id
        if not model_dir.exists():
            raise FileNotFoundError(f"找不到模型: {model_dir}")
        
        with open(model_dir / "meta.json", 'r', encoding='utf-8') as f:
            meta = json.load(f)
            
        cls_name = meta['model_class']
        
        # 简单工厂模式
        if cls_name == 'LGBMClassifierModel':
            model = LGBMClassifierModel()
        elif cls_name == 'XGBoostModel':
            model = XGBoostModel()
        elif cls_name == 'CatBoostModel':
            model = CatBoostModel()
        else:
            raise ValueError(f"未知模型类型: {cls_name}")
            
        model.load(str(model_dir / "model.pkl"))
        return model, meta

    def get_latest_id(self, experiment_prefix):
        '''
        自动寻找最新的模型ID
        :param experiment_prefix: 例如 "n_pattern/v1_lgbm"
        '''
        # 1. 拆分路径和前缀
        parts = experiment_prefix.split('/')
        if len(parts) > 1:
            search_dir = self.root.joinpath(*parts[:-1]) # model_zoo/n_pattern
            prefix = parts[-1] # v1_lgbm
        else:
            search_dir = self.root
            prefix = experiment_prefix

        if not search_dir.exists():
            print(f"⚠️ 目录不存在: {search_dir}")
            return None

        # 2. 扫描目录下所有以 prefix 开头的文件夹
        candidates = []
        for p in search_dir.iterdir():
            if p.is_dir() and p.name.startswith(prefix):
                candidates.append(p)
        
        if not candidates:
            print(f"⚠️ 未找到前缀为 {prefix} 的模型")
            return None

        # 3. 按文件夹名排序 (时间戳在最后，且格式为 YYYYMMDD_HHMMSS，可以直接字符串排序)
        candidates.sort(key=lambda x: x.name, reverse=True)
        latest_dir = candidates[0]
        
        # 返回相对路径 ID
        relative_id = str(latest_dir.relative_to(self.root)).replace("\\", "/")
        print(f"🔎 自动锁定最新模型: {relative_id}")
        return relative_id