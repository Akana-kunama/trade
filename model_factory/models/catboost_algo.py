from catboost import CatBoostClassifier
import joblib
from ..interface import BaseModel

class CatBoostModel(BaseModel):
    def __init__(self, params=None):
        default_params = {
            'iterations': 1000,
            'learning_rate': 0.05,
            'loss_function': 'Logloss',
            'eval_metric': 'AUC',
            'verbose': False,
            'random_seed': 42,
            'allow_writing_files': False
        }
        if params:
            default_params.update(params)
        self.params = default_params
        self.model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.model = CatBoostClassifier(**self.params)
        self.model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val) if X_val is not None else None,
            early_stopping_rounds=50
        )
        return self

    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]

    def save(self, path):
        # CatBoost 自带 save_model 但 joblib 更通用
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)