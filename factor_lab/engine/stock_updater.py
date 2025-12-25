import os
import json
import pandas as pd
import datetime
from tqdm import tqdm
from .registry import get_all_factor_classes


class StockFactorUpdater:
    def __init__(self, raw_data_dir, factor_store_dir):
        self.raw_data_dir = raw_data_dir
        self.factor_store_dir = factor_store_dir

        storage_root = os.path.dirname(factor_store_dir.rstrip(os.sep))
        self.meta_path = os.path.join(storage_root, "metadata.json")
        
        self.metadata = self._load_metadata()
        self.factors = get_all_factor_classes() 
        
        os.makedirs(factor_store_dir, exist_ok=True)

    def _load_metadata(self):
        if os.path.exists(self.meta_path):
            with open(self.meta_path, 'r') as f:
                return json.load(f)
        return {}

    def _save_metadata(self):
        with open(self.meta_path, 'w') as f:
            json.dump(self.metadata, f, indent=4)

    def process_single_stock(self, stock_code, force_update=False):
        '''处理单只股票：增量更新'''
        raw_path = os.path.join(self.raw_data_dir, f"{stock_code}.parquet")
        store_path = os.path.join(self.factor_store_dir, f"{stock_code}.parquet")
        
        if not os.path.exists(raw_path):
            return 
            
        # 1. 读取原始数据
        df_raw = pd.read_parquet(raw_path)
        if df_raw.empty:
            return
        
        # 确保索引是 datetime
        if not isinstance(df_raw.index, pd.DatetimeIndex):
            df_raw.index = pd.to_datetime(df_raw.index)
        df_raw = df_raw.sort_index()
        
        latest_raw_date = df_raw.index.max().strftime('%Y-%m-%d')
        last_update_date = self.metadata.get(stock_code, "1990-01-01")
        
        # 2. 检查是否需要更新
        if not force_update and latest_raw_date <= last_update_date:
            return # 数据已最新，跳过
            
        # 3. 确定计算区间
        # 为了保证rolling(N)计算正确，需要往前多读一点数据 (buffer)
        # 这里简化处理：如果是增量，读取全部raw计算后再截取；
        # 优化方案：只取 last_update_date - 60 days 之后的数据进行计算
        
        # 计算因子 DataFrame
        df_factors = pd.DataFrame(index=df_raw.index)
        
        for name, factor_obj in self.factors.items():
            try:
                df_factors[name] = factor_obj.compute(df_raw)
            except Exception as e:
                # print(f"Error calculating {name} for {stock_code}: {e}")
                df_factors[name] = None

        # 4. 存储 (覆盖式存储最简单，Append稍微复杂，这里先用覆盖以保证数据一致性)
        # 如果Parquet文件很大，才考虑 append 模式
        # 将原始的 open, close 也放进去方便后续对其
        df_final = df_raw[['open', 'high', 'low', 'close', 'volume']].join(df_factors)
        
        df_final.to_parquet(store_path)
        
        # 5. 更新元数据
        self.metadata[stock_code] = latest_raw_date

    def run_batch(self, stock_codes=None):
        '''批量运行'''
        if stock_codes is None:
            # 默认扫描目录下所有文件
            files = os.listdir(self.raw_data_dir)
            stock_codes = [f.replace('.parquet', '') for f in files if f.endswith('.parquet')]
        
        print(f"开始更新因子库，共 {len(stock_codes)} 只股票...")
        
        # 批量处理并保存元数据
        count = 0
        for code in tqdm(stock_codes):
            try:
                self.process_single_stock(code)
                count += 1
                if count % 100 == 0:
                    self._save_metadata()
            except Exception as e:
                print(f"Error processing {code}: {e}")
                
        self._save_metadata()
        print("因子库更新完成。")