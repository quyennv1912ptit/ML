from typing import Dict

import numpy as np
from Node import Node

class CARTRegressor:
    def __init__(self, max_depth = None, max_features = None, min_samples_split = 2, min_samples_leaf = 1, min_mse_decrease = 0):
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_mse_decrease = min_mse_decrease
        self.root = None
        self.n_leaves = 0

    @staticmethod
    def mse(y):
        if len(y) == 0:
            return 0
        return np.var(np.asarray(y))

    def best_split(self, X, y):
        n_samples, n_features = X.shape
        S1_total, S2_total = np.sum(y), np.sum(y ** 2)
        parent_mse = self.mse(y)
        best: Dict | None = None
        if self.max_features is None:
            n_sub_features = n_features
        elif self.max_features == 'sqrt':
            n_sub_features = int(np.sqrt(n_features))
        elif self.max_features == 'third':
            n_sub_features = max(1, int(n_features / 3))
        elif isinstance(self.max_features, int):
            n_sub_features = min(self.max_features, n_features)
        elif isinstance(self.max_features, float):
            n_sub_features = max(1, int(self.max_features * n_features))
        else:
            n_sub_features = n_features

        feature_indices = np.random.choice(n_features, n_sub_features, replace=False)

        for j in feature_indices:
            col = X[:, j]
            order = np.argsort(col, kind='mergesort')
            col_sorted = col[order]
            y_sorted = y[order]

            S1_left, S2_left = 0, 0
            S1_right, S2_right = S1_total, S2_total
            for i in range(n_samples - 1):
                S1_left += y_sorted[i]
                S2_left += y_sorted[i] ** 2
                S1_right -= y_sorted[i]
                S2_right -= y_sorted[i] ** 2
                n_left = i + 1
                n_right = n_samples - n_left

                if col_sorted[i] == col_sorted[i+1]:
                    continue
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                mse_left = S2_left / n_left - (S1_left  / n_left) ** 2
                mse_right = S2_right / n_right - (S1_right / n_right) ** 2

                weighted = (n_left / n_samples) * mse_left + (n_right / n_samples) * mse_right

                gain = parent_mse - weighted

                if best is None or gain > best['gain']:
                    threshold = (col_sorted[i] + col_sorted[i+1]) / 2.0
                    best = {'feature': j, 'threshold': threshold, 'gain': gain}

        return best

    def build(self, X, y, depth):
        node = Node(depth)
        n_samples = len(y)
        node.n_samples = n_samples
        node.impurity = self.mse(y)

        stop = (
            node.impurity == 0 or
            n_samples < self.min_samples_split or
            (self.max_depth is not None and depth >= self.max_depth)
        )

        split: Dict | None = None
        if not stop:
            split = self.best_split(X, y)
            if split is None or split['gain'] < self.min_mse_decrease:
                stop = True

        if stop or split is None:
            node.value = np.mean(y)
            self.n_leaves += 1
            return node

        j, t = split['feature'], split['threshold']
        left = X[:, j] <= t

        node.feature = j
        node.threshold = t

        node.left = self.build(X[left], y[left], depth+1)
        node.right = self.build(X[~left], y[~left], depth+1)
        return node

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.n_leaves = 0
        self.root = self.build(X, y, 0)
        return self

    def predict_one(self, x, node):
        while not node.is_leaf():
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.value

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self.predict_one(x, self.root) for x in X])

    def depth(self):
        def d(node: Node|None):
            if node is None:
                return 0
            if node.is_leaf():
                return node.depth
            return max(d(node.left), d(node.right))
        return d(self.root)
