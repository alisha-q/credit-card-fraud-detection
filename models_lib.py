"""Shared model class definitions so pickled reference models can be unpickled
consistently across scripts (train scripts, evaluation, SHAP, and the Streamlit app)."""
import numpy as np

class LogisticRegressionNP:
    def __init__(self, lr=0.1, n_iter=2000, l2=1e-3):
        self.lr=lr; self.n_iter=n_iter; self.l2=l2
    def fit(self, X, y):
        n, d = X.shape
        self.w = np.zeros(d); self.b = 0.0
        for i in range(self.n_iter):
            z = X @ self.w + self.b
            p = 1/(1+np.exp(-z))
            grad_w = X.T @ (p - y) / n + self.l2*self.w
            grad_b = np.mean(p - y)
            self.w -= self.lr*grad_w
            self.b -= self.lr*grad_b
        return self
    def predict_proba(self, X):
        z = X @ self.w + self.b
        return 1/(1+np.exp(-z))
