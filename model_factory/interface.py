from abc import ABC, abstractmethod
import pandas as pd

class BaseModel(ABC):
    '''所有模型必须继承的基类，保证对外接口一致'''
    
    @abstractmethod
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        '''训练接口'''
        pass

    @abstractmethod
    def predict(self, X):
        '''预测接口，返回分数或概率'''
        pass
        
    @abstractmethod
    def save(self, path):
        '''保存模型二进制文件'''
        pass
        
    @abstractmethod
    def load(self, path):
        '''加载模型二进制文件'''
        pass