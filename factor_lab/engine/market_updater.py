import os
import pandas as pd
import glob
from tqdm import tqdm

class MarketFactorUpdater:
    '''
    专门计算全市场截面因子（如涨停家数、市场情绪）
    这些因子不是基于单只股票的，而是基于全市场当日表现汇总的。
    '''
    def __init__(self, raw_data_dir, output_path):
        self.raw_data_dir = raw_data_dir
        self.output_path = output_path

    def run(self):
        print("正在计算全市场情绪指标...")
        file_pattern = os.path.join(self.raw_data_dir, "*.parquet")
        all_files = glob.glob(file_pattern)
        
        processed_dfs = []
        
        # 这是一个IO密集型操作，每日运行一次即可
        for file_path in tqdm(all_files, desc="Market Sentiment"):
            try:
                df = pd.read_parquet(file_path, columns=['close', 'limit_up', 'limit_down'])
                if df.empty: continue
                
                # 简单计算
                epsilon = 1e-4
                # 兼容不同数据源列名
                if 'limit_up' in df.columns:
                    df['is_limit_up'] = (df['close'] >= (df['limit_up'] - epsilon)).astype(int)
                    df['is_limit_down'] = (df['close'] <= (df['limit_down'] + epsilon)).astype(int)
                else:
                    # 如果没有涨跌停列，这里暂时跳过或做估算
                    continue
                
                processed_dfs.append(df[['is_limit_up', 'is_limit_down']])
            except:
                continue
        
        if not processed_dfs:
            print("无有效数据计算市场因子")
            return

        # 合并大数据表 (内存消耗点，注意)
        all_data = pd.concat(processed_dfs)
        
        # 按日期聚合
        daily_stats = all_data.groupby(all_data.index).agg({
            'is_limit_up': 'sum',
            'is_limit_down': 'sum'
        })
        
        daily_stats.rename(columns={
            'is_limit_up': 'market_limit_up_count',
            'is_limit_down': 'market_limit_down_count'
        }, inplace=True)
        
        # 保存为CSV (市场因子一般较小，CSV方便查看)
        daily_stats.to_csv(self.output_path)
        print(f"市场因子已保存至: {self.output_path}")