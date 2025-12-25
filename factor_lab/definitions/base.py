from abc import ABC, abstractmethod
import pandas as pd

class BaseFactor(ABC):
    name = ""           # 因子在Parquet中的列名
    description = ""    # 因子描述
    
    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.Series:
        '''
        输入: 个股的原始行情数据 (Open, High, Low, Close, Volume)
        输出: 因子值 Series (Index必须与df对齐)
        '''
        pass