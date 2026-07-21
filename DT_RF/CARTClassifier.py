import numpy as np
from Node import Node
from typing import Dict

def gini(counts):
    n = np.sum(counts)
    if n == 0:
        return 0
    p = counts / n
    return 1 - np.sum(p ** 2)

def entropy(counts):
    n = np.sum(counts)
    if n == 0:
        return 0
    p = counts / n
    p = p[p > 0]
    log_p = np.log2(p)
    return -np.sum(p  * log_p)

IMPURITY_FUNC = {'gini': gini, 'entropy': entropy}

class CARTClassifier:
    def __init__(self, criterion='gini', max_depth = None, max_features = None, min_samples_split = 2, min_samples_leaf = 1, ccp_alpha = 0, min_impurity_decrease = 0):
        self.criterion = criterion
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.root = None
        self.n_classes_ = 0
        self.n_leaves_ = 0
        self._impurity_func = IMPURITY_FUNC[criterion]
        self.n_total_samples = 0
        self.feature_importances = None
        self.ccp_alpha = ccp_alpha

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=int)
        n_samples, n_features  = X.shape
        self.n_total_samples = n_samples
        self.feature_importances = np.zeros(n_features)
        self.n_classes_ = int(max(y)) + 1
        self.n_leaves_ = 0
        self.root = self._build(X, y, depth = 0)
        total = self.feature_importances.sum()
        if total > 0:
            self.feature_importances /= total
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_one(x, self.root) for x in X])

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis = 1)

    def _build(self, X, y, depth):
        node = Node(depth)
        n_samples = len(y)
        node.n_samples = n_samples
        counts = np.bincount(y, minlength=self.n_classes_)
        node.impurity = self._impurity_func(counts)
        stop = (
            node.impurity == 0 or
            n_samples < self.min_samples_split or
            (self.max_depth is not None and depth >= self.max_depth)
        )
        split: Dict | None = None
        if not stop:
            split = self. _best_split(X, y, counts)
            if split is None or split['gain'] <= self.min_impurity_decrease:
                stop = True

        if stop or split is None:
            node.value = counts / n_samples
            self.n_leaves_ += 1
            return node
        if self.feature_importances is not None:
            self.feature_importances[split['feature']] += (n_samples / self.n_total_samples) * split['gain']

        j, t = split['feature'], split['threshold']

        left = X[:, j] <= t
        node.feature = j
        node.threshold = t
        node.left = self._build(X[left], y[left], depth+1)
        node.right = self._build(X[~left], y[~left], depth+1)
        return node

    def _best_split(self, X, y, parent_counts):
        n_samples, n_features = X.shape
        parent_impurity = self._impurity_func(parent_counts)
        best = None

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

            left_counts = np.zeros(self.n_classes_)
            right_counts = parent_counts.copy()

            for i in range(n_samples - 1):
                c = y_sorted[i]
                left_counts[c] += 1
                right_counts[c] -= 1
                n_left = i+1
                n_right = n_samples - n_left

                # Nếu đặc trưng tại hai điểm có giá trị giống nhau
                if col_sorted[i] == col_sorted[i+1]:
                    continue
                # Nếu nhánh trái hoặc nhánh phải không đủ số phân tử tối thiểu
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                imp_left = self._impurity_func(left_counts)
                imp_right = self._impurity_func(right_counts)

                weighted = (n_left / n_samples) * imp_left + (n_right / n_samples) * imp_right

                gain = parent_impurity - weighted

                if best is None or gain > best['gain']:
                    threshold = (col_sorted[i] + col_sorted[i+1]) / 2
                    best = {'feature': j, 'threshold': threshold, 'gain': gain}

        return best

    def prune(self, node):
        if node is None:
            return 0.0, 0

        if node.is_leaf:
            cost = (node.n_samples / self.n_total_samples) * node.impurity
            return cost, 1

        cost_left, leaves_left = self.prune(node.left)
        cost_right, leaves_right = self.prune(node.right)

        cost_children = cost_left + cost_right
        leaves_children = leaves_left + leaves_right

        cost_node = (node.n_samples / self.n_total_samples) * node.impurity

        if cost_node <= cost_children + self.ccp_alpha * (leaves_children - 1):
            node.left = None
            node.right = None
            self.n_leaves_ -= (leaves_children - 1)
            return cost_node, 1

        return cost_children, leaves_children

    def _predict_one(self, x, node):
        while not node.is_leaf():
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.value

    def depth(self):
        def _d(node):
            if node.is_leaf():
                return node.depth
            return max(_d(node.left), _d(node.right))
        return _d(self.root)
