import pandas as pd
import numpy as np
from .base import BaseFactor

class MA5(BaseFactor):
    name = "ma5"
    description = "5日收盘均线"
    def compute(self, df):
        return df['close'].rolling(window=5).mean()

class VolRatio(BaseFactor):
    name = "vol_ratio"
    description = "量比: 当日成交量/5日均量"
    def compute(self, df):
        ma_vol_5 = df['volume'].rolling(window=5).mean()
        return df['volume'] / ma_vol_5.replace(0, np.nan)

class UpperShadow(BaseFactor):
    name = "upper_shadow"
    description = "上影线长度: (High - Max(Open, Close)) / Close"
    def compute(self, df):
        body_top = df[['open', 'close']].max(axis=1)
        return (df['high'] - body_top) / df['close']

class BodySize(BaseFactor):
    name = "body_size"
    description = "实体大小: abs(Close-Open)/Open"
    def compute(self, df):
        return abs(df['close'] - df['open']) / df['open']

class MA5Bias(BaseFactor):
    name = "ma5_bias"
    description = "5日均线乖离率"
    def compute(self, df):
        ma5 = df['close'].rolling(window=5).mean()
        return (df['close'] - ma5) / ma5

class HighLowRatio(BaseFactor):
    name = "high_low_ratio"
    description = "高低波幅: (High-Low)/PreClose"
    def compute(self, df):
        pre_close = df['close'].shift(1)
        return (df['high'] - df['low']) / pre_close

class IsLimitUp(BaseFactor):
    name = "is_limit_up"
    description = "是否涨停 (1/0)"
    def compute(self, df):
        # 兼容处理：如果没有limit_up列，简单估算
        if 'limit_up' in df.columns:
            limit = df['limit_up']
        else:
            limit = df['close'].shift(1) * 1.1 # 简易兜底
        
        epsilon = 1e-4
        return (df['close'] >= (limit - epsilon)).astype(int)