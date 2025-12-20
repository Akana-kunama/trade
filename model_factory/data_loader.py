import pandas as pd
import sys
from pathlib import Path

# 尝试加载配置
try:
    from config.path_config import FACTOR_STORE_DIR, MARKET_FACTOR_PATH
except ImportError:
    pass

from factor_lab.service.data_provider import FeatureService

class DatasetLoader:
    def __init__(self):
        self.svc = FeatureService(
            factor_store_dir=str(FACTOR_STORE_DIR),
            market_factor_path=str(MARKET_FACTOR_PATH)
        )

    def load_dataset(self, config):
        print(f"🔄 [Loader] 请求特征: {len(config['features'])} 个, 目标: {config['label']}")
        
        # 1. 组装列
        required_cols = list(set(config['features'] + [config['label']]))
        filter_cfg = config.get('filter')
        if filter_cfg:
            required_cols.append(filter_cfg['column'])
            
        # 2. 调用服务层取数
        df = self.svc.get_features(
            codes=config.get('codes'),
            factor_names=required_cols,
            start_date=config['start_date'],
            end_date=config['end_date']
        )
        
        if df.empty:
            print("⚠️ [Loader] 加载数据为空！")
            return pd.DataFrame()

        # ==========================================
        # 🛡️ [核心修复] 暴力统一日期列名
        # ==========================================
        # 1. 先把索引重置，让日期变成普通列
        #    reset_index 会把 index 变成一列，名字通常叫 'index' 或原名 'datetime'
        df = df.reset_index()
        
        # 2. 检查并重命名
        #    通常 parquet 读取后 index 名字叫 'datetime' 或者 'index'
        #    我们要把它统一改成 'date'
        cols_lower = {c.lower(): c for c in df.columns}
        
        if 'date' in cols_lower:
            # 已经有 date 列了 (可能是 reset_index 出来的，也可能是原本就有的)
            # 确保列名是全小写 'date'
            real_name = cols_lower['date']
            if real_name != 'date':
                df.rename(columns={real_name: 'date'}, inplace=True)
        elif 'datetime' in cols_lower:
            # 如果叫 datetime，改名叫 date
            real_name = cols_lower['datetime']
            df.rename(columns={real_name: 'date'}, inplace=True)
        elif 'index' in df.columns:
            # 如果叫 index，改名叫 date
            df.rename(columns={'index': 'date'}, inplace=True)
            
        # 3. 最终检查
        if 'date' not in df.columns:
            print(f"❌ [Loader] 无法识别日期列！当前列名: {df.columns.tolist()}")
            return pd.DataFrame()
            
        # 4. 强制转为 datetime 格式 (防止是字符串)
        df['date'] = pd.to_datetime(df['date'])
        # ==========================================

        # 5. 应用过滤器
        if filter_cfg:
            col = filter_cfg['column']
            val = filter_cfg['value']
            if col in df.columns:
                df = df[df[col] == val].copy()
            
        # 6. 缺失值填充
        for feat in config['features']:
            if feat in df.columns:
                df[feat] = df[feat].fillna(df[feat].mean())
                
        df = df.dropna(subset=[config['label']])
        
        return df