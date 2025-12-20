import pandas as pd
import numpy as np
from .base import BaseFactor

class NextOpenChange(BaseFactor):
    name = "next_open_change"
    description = "T+1开盘涨幅 (Label)"
    def compute(self, df):
        next_open = df['open'].shift(-1)
        return (next_open - df['close']) / df['close']

class StrategyLabel(BaseFactor):
    name = "label_strategy"
    description = "符合策略条件的Label (1/0)"
    def compute(self, df):
        # 获取未来数据
        next_close = df['close'].shift(-1)
        next_open = df['open'].shift(-1)
        next_high = df['high'].shift(-1)
        next2_open = df['open'].shift(-2) # T+2

        # 逻辑复刻
        # 1. 收阳
        cond_1 = next_close > next_open
        # 2. 冲高 > 3%
        cond_2 = (next_high - next_open) / next_open > 0.03
        # 3. T+2 开盘不深跌 (Open(T+2) >= Close(T+1))
        # 注意：这里需要处理除0或NaN
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = next2_open / next_close
            cond_3 = ratio >= 1
        
        # 综合
        final_cond = cond_1 & cond_2 & cond_3
        return final_cond.fillna(False).astype(int)