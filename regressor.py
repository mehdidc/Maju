from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.decomposition import KernelPCA
from sklearn import neighbors
from sklearn.decomposition import PCA

class Regressor(BaseEstimator):
    def __init__(self):
        self.clf = Pipeline([('scaler', StandardScaler()),
                              ("PCA", PCA(n_components=80)),
                              ("GB",GradientBoostingRegressor(n_estimators = 500, max_depth=8))])
    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)
