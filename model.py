from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor


class Model:
    def __init__(self):
        parameters = {
            'lambda': 0.02202645400480881,
            'learning_rate': 0.15980545084240536,
            'max_depth': 11,
            'gamma': 1.091827352641474,
        }
        self.model = XGBRegressor(**parameters)

    def fit(self, X, y):
        self.model = self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


