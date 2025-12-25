# factor_lab/definitions/strategy_n_pattern.py
import pandas as pd
import numpy as np
from .base import BaseFactor

class NPatternSignal(BaseFactor):
    name = "n_pattern_signal"
    description = "N字反转选股信号 (1=入选, 0=落选)"
    
    def compute(self, df):
        # --- 1. 基础数据准备 ---
        close = df['close']
        open_ = df['open']
        # 简易涨停判断 (如果有 limit_up 列最好，没有则估算)
        if 'limit_up' in df.columns:
            is_limit_up = (close >= (df['limit_up'] - 0.0001))
        else:
            # 兜底逻辑：涨幅>9.5%视作涨停
            is_limit_up = (close / close.shift(1) - 1) > 0.098
            
        # 过去60天涨停数 (shift(1)代表不含今日)
        recent_limit_ups_60 = is_limit_up.shift(1).rolling(window=60).sum()
        
        # --- 2. 辅助变量 ---
        L1 = is_limit_up.shift(1).fillna(False) # 昨天涨停
        L2 = is_limit_up.shift(2).fillna(False) # 前天涨停
        L3 = is_limit_up.shift(3).fillna(False)
        
        open_T1 = open_.shift(1)
        open_T2 = open_.shift(2)
        open_T3 = open_.shift(3)
        
        cur_limit_up = is_limit_up # 今天是否涨停
        
        # --- 3. 筛选逻辑 ---
        # 过热过滤
        not_overheated = (recent_limit_ups_60 < 6)
        
        # 场景 1: T-1涨停，T未涨停
        # shift(2) means limits before the pattern started
        past_counts_1 = is_limit_up.shift(2).rolling(6).sum()
        case_1 = (L1) & (~cur_limit_up) & (past_counts_1 < 2) & (close >= open_T1)
        
        # 场景 2: T-2涨停，T-1, T未涨停
        past_counts_2 = is_limit_up.shift(3).rolling(6).sum()
        case_2 = (L2) & (~L1) & (~cur_limit_up) & (past_counts_2 < 2) & (close >= open_T2)
        
        # 场景 3: T-3涨停，T-2, T-1, T未涨停
        past_counts_3 = is_limit_up.shift(4).rolling(6).sum()
        case_3 = (L3) & (~L2) & (~L1) & (~cur_limit_up) & (past_counts_3 < 2) & (close >= open_T3)
        
        # 综合
        final_signal = (case_1 | case_2 | case_3) & not_overheated
        
        return final_signal.fillna(False).astype(int)