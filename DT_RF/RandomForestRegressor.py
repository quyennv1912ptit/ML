from CARTRegressor import CARTRegressor
import numpy as np

class RandomForestRegressor:
    def __init__(self, n_estimators = 100, max_depth = None, max_features = None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        n_samples = X.shape[0]
        for _ in range(self.n_estimators):
            bootstrap_indices = np.random.choice(a=n_samples, size=n_samples, replace=True)

            X_bootstrap = X[bootstrap_indices]
            y_bootstrap = y[bootstrap_indices]

            tree = CARTRegressor(max_depth = self.max_depth, max_features = self.max_features)

            tree.fit(X_bootstrap, y_bootstrap)

            self.trees.append(tree)

    def predict(self, X):
        tree_predicts = np.array([tree.predict(X) for tree in self.trees]).T
        return np.mean(tree_predicts, axis=1)
