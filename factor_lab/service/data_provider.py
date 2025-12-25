import pandas as pd
import os

class FeatureService:
    def __init__(self, factor_store_dir, market_factor_path=None):
        self.factor_store_dir = factor_store_dir
        self.market_factor_path = market_factor_path
        self._market_cache = None

    def _load_market_data(self):
        if self._market_cache is None and self.market_factor_path and os.path.exists(self.market_factor_path):
            df = pd.read_csv(self.market_factor_path, index_col=0, parse_dates=True)
            self._market_cache = df
        return self._market_cache

    def get_features(self, codes: list, factor_names: list = None, start_date=None, end_date=None):
        '''
        对外统一接口：获取训练/预测数据
        '''
        result_list = []
        market_df = self._load_market_data()
        
        for code in codes:
            path = os.path.join(self.factor_store_dir, f"{code}.parquet")
            if not os.path.exists(path):
                continue
            
            try:
                # 1. 基础读取 (Parquet 列裁剪加速)
                # 必须包含 date 索引，以及请求的因子
                df = pd.read_parquet(path)
                
                # 时间过滤
                if start_date:
                    df = df[df.index >= start_date]
                if end_date:
                    df = df[df.index <= end_date]
                
                # 2. 列过滤
                if factor_names:
                    # 确保请求的列存在，防止报错
                    valid_cols = [c for c in factor_names if c in df.columns]
                    # 保留OHLCV如果是必须的
                    # df = df[valid_cols] 
                    # 这里暂时返回所有列，或者你可以只选 valid_cols
                
                # 3. 合并市场因子
                if market_df is not None:
                    df = df.join(market_df, how='left')
                
                df['code'] = code
                result_list.append(df)
                
            except Exception as e:
                print(f"Error loading {code}: {e}")
                continue
        
        if result_list:
            return pd.concat(result_list)
        return pd.DataFrame()