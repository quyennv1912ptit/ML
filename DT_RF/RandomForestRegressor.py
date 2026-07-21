from CARTRegressor import CARTRegressor
import numpy as np

class RandomForestRegressor:
    def __init__(self, n_estimators = 100, max_depth = None, max_features = None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.trees = []
        self.feature_importances = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.trees = []
        n_samples, n_features = X.shape
        self.feature_importances = np.zeros(n_features)
        oob_sums = np.zeros(n_samples)
        oob_counts = np.zeros(n_samples)

        for _ in range(self.n_estimators):
            bootstrap_indices = np.random.choice(a=n_samples, size=n_samples, replace=True)

            X_bootstrap = X[bootstrap_indices]
            y_bootstrap = y[bootstrap_indices]

            tree = CARTRegressor(max_depth = self.max_depth, max_features = self.max_features)

            tree.fit(X_bootstrap, y_bootstrap)
            if tree.feature_importances is not None:
                self.feature_importances += tree.feature_importances

            self.trees.append(tree)

            in_bag = np.zeros(n_samples, dtype=bool)
            in_bag[bootstrap_indices] = True
            oob_idx = np.where(~in_bag)[0]
            for idx, p in zip(oob_idx, tree.predict(X[oob_idx])):
                oob_sums[idx] += p
                oob_counts[idx] += 1

        self.feature_importances /= self.n_estimators
        has_oob = oob_counts > 0
        if len(has_oob) > 0:
            y_pred = oob_sums[has_oob] / oob_counts[has_oob]
            y_true = y[has_oob]
            ss_total = np.sum((y_true - y_true.mean()) ** 2)
            ss_res = np.sum((y_true - y_pred) ** 2)
            self.oob_score = 1 - ss_res / ss_total

    def predict(self, X):
        tree_predicts = np.array([tree.predict(X) for tree in self.trees]).T
        return np.mean(tree_predicts, axis=1)
