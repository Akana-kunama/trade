import xgboost as xgb
import joblib
from ..interface import BaseModel

class XGBoostModel(BaseModel):
    def __init__(self, params=None):
        self.params = params if params else {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'n_estimators': 1000,
            'learning_rate': 0.05,
            'max_depth': 5,
            'n_jobs': -1,
            'random_state': 42,
            'early_stopping_rounds': 50
        }
        self.model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.model = xgb.XGBClassifier(**self.params)
        eval_set = [(X_val, y_val)] if X_val is not None else None
        
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )
        return self

    def predict(self, X):
        # 返回概率值 (class 1)
        return self.model.predict_proba(X)[:, 1]

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)