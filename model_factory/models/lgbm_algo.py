import lightgbm as lgb
import joblib
from ..interface import BaseModel

class LGBMClassifierModel(BaseModel):
    def __init__(self, params=None):
        # 默认参数
        self.params = params if params else {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'n_jobs': -1,
            'verbose': -1,
            'learning_rate': 0.05,
            'num_leaves': 31
        }
        self.model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_sets = [train_data]
        if X_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)
            
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=500,
            valid_sets=valid_sets,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=50)
            ]
        )
        return self

    def predict(self, X):
        if self.model is None:
            raise ValueError("模型尚未训练！")
        return self.model.predict(X)

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)