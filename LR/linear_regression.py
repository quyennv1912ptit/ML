import numpy as np

class LinearRegression:
    def __init__(self):
        self.w = None

    @staticmethod
    def add_bias(X):
        n_samples = X.shape[0]
        return np.hstack([np.ones((n_samples, 1)), X])

    @staticmethod
    def mse(y, y_pred):
        return np.mean((y-y_pred)**2)

    @staticmethod
    def r_square(y, y_pred):
        ss_res = np.sum((y - y_pred) ** 2)
        ss_total = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_total)

    def fit_normal_equation(self, X, y):
        Xb = self.add_bias(X)
        self.w = np.linalg.inv(Xb.T @ Xb)@Xb.T@y

    def fit_gradient_descent(self, X, y, n_iters = 100, lr = 0.02):
        Xb = self.add_bias(X)
        n_samples = X.shape[0]
        self.w = np.zeros(Xb.shape[1])
        for _ in range(n_iters):
            gradient = (2 / n_samples) * Xb.T @ (Xb @ self.w - y)
            self.w = self.w - lr * gradient

    def predict(self, X):
        return self.add_bias(X) @ self.w
