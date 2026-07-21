import numpy as np
from CARTClassifier import CARTClassifier

class RandomForestClassifier:
    def __init__(self, n_estimators = 100, max_depth = None, max_features = 'sqrt'):
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
        self.n_classes = int(np.max(y)) + 1
        self.feature_importances = np.zeros(n_features)
        # số phiếu bầu mẫu i thuộc class c, tính gộp qua tất cả các cây
        oob_votes = np.zeros((n_samples, self.n_classes))
        # tổng số phiếu bầu của mỗi mẫu i (Số cây mà mẫu i thuộc tập OOB)
        oob_counts = np.zeros(n_samples)
        for _ in range(self.n_estimators):
            bootstrap_indices = np.random.choice(a=n_samples, size=n_samples, replace=True)

            X_bootstrap = X[bootstrap_indices]
            y_bootstrap = y[bootstrap_indices]

            tree = CARTClassifier(max_depth = self.max_depth, max_features = self.max_features)

            tree.fit(X_bootstrap, y_bootstrap)
            if tree.feature_importances is not None:
                self.feature_importances += tree.feature_importances

            self.trees.append(tree)

            in_bag = np.zeros(n_samples, dtype=bool)
            in_bag[bootstrap_indices] = True
            oob_idx = np.where(~in_bag)[0]
            if len(oob_idx) > 0:
                preds = tree.predict(X[oob_idx])
                for idx, p in zip(oob_idx, preds):
                    oob_votes[idx, p] += 1
                    oob_counts[idx] += 1

        self.feature_importances /= self.n_estimators

        has_oob = oob_counts > 0
        oob_pred = np.argmax(oob_votes[has_oob], axis=1)
        self.oob_score = np.mean(oob_pred == y[has_oob])
        self.oob_coverage = has_oob.mean()

    def predict(self, X):
        tree_predicts = np.array([tree.predict(X) for tree in self.trees]).T
        return np.array([np.argmax(np.bincount(pred)) for pred in tree_predicts])
