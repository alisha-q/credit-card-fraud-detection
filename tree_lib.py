"""Lightweight numpy-only CART implementation used for the Random Forest and
Gradient Boosting reference models (stand-ins for sklearn.RandomForestClassifier
and xgboost.XGBClassifier, which weren't installable in the build sandbox)."""
import numpy as np

class Node:
    __slots__ = ("feature","threshold","left","right","value")
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature=feature; self.threshold=threshold; self.left=left; self.right=right; self.value=value

def _gini(y):
    if len(y)==0: return 0
    p = np.mean(y)
    return 1 - p**2 - (1-p)**2

def _mse(y):
    if len(y) == 0: return 0
    return np.mean((y - y.mean())**2)

def _best_split(X, y, n_features_try, criterion, rng, n_quantiles=16):
    n, d = X.shape
    feat_idx = rng.choice(d, size=n_features_try, replace=False)
    best_gain, best_feat, best_thr = -np.inf, None, None
    parent_impurity = criterion(y)
    for f in feat_idx:
        col = X[:, f]
        qs = np.unique(np.quantile(col, np.linspace(0.1, 0.9, n_quantiles)))
        for thr in qs:
            left_mask = col <= thr
            if left_mask.sum() < 5 or (~left_mask).sum() < 5:
                continue
            yl, yr = y[left_mask], y[~left_mask]
            w = len(yl)/n
            impurity = w*criterion(yl) + (1-w)*criterion(yr)
            gain = parent_impurity - impurity
            if gain > best_gain:
                best_gain, best_feat, best_thr = gain, f, thr
    return best_feat, best_thr, best_gain

def _build_tree(X, y, depth, max_depth, min_samples_split, n_features_try, criterion, rng, leaf_fn):
    if depth >= max_depth or len(y) < min_samples_split or len(np.unique(y)) == 1:
        return Node(value=leaf_fn(y))
    feat, thr, gain = _best_split(X, y, n_features_try, criterion, rng)
    if feat is None or gain <= 1e-12:
        return Node(value=leaf_fn(y))
    mask = X[:, feat] <= thr
    left = _build_tree(X[mask], y[mask], depth+1, max_depth, min_samples_split, n_features_try, criterion, rng, leaf_fn)
    right = _build_tree(X[~mask], y[~mask], depth+1, max_depth, min_samples_split, n_features_try, criterion, rng, leaf_fn)
    return Node(feature=feat, threshold=thr, left=left, right=right)

def _predict_row(node, x):
    while node.value is None:
        node = node.left if x[node.feature] <= node.threshold else node.right
    return node.value

class DecisionTreeClassifierNP:
    def __init__(self, max_depth=6, min_samples_split=20, n_features_try=None, seed=0):
        self.max_depth=max_depth; self.min_samples_split=min_samples_split
        self.n_features_try=n_features_try; self.seed=seed
    def fit(self, X, y):
        rng = np.random.RandomState(self.seed)
        d = X.shape[1]
        nft = self.n_features_try or max(1, int(np.sqrt(d)))
        leaf_fn = lambda yy: np.mean(yy) if len(yy) else 0.0
        self.root = _build_tree(X, y, 0, self.max_depth, self.min_samples_split, nft, _gini, rng, leaf_fn)
        return self
    def predict_proba(self, X):
        return np.array([_predict_row(self.root, x) for x in X])

class DecisionTreeRegressorNP:
    """Used inside gradient boosting to fit residuals/gradients."""
    def __init__(self, max_depth=3, min_samples_split=20, n_features_try=None, seed=0):
        self.max_depth=max_depth; self.min_samples_split=min_samples_split
        self.n_features_try=n_features_try; self.seed=seed
    def fit(self, X, y):
        rng = np.random.RandomState(self.seed)
        d = X.shape[1]
        nft = self.n_features_try or d
        leaf_fn = lambda yy: np.mean(yy) if len(yy) else 0.0
        self.root = _build_tree(X, y, 0, self.max_depth, self.min_samples_split, nft, _mse, rng, leaf_fn)
        return self
    def predict(self, X):
        return np.array([_predict_row(self.root, x) for x in X])


class RandomForestClassifierNP:
    def __init__(self, n_estimators=25, max_depth=6, min_samples_split=20, n_features_try=None, seed=42):
        self.n_estimators=n_estimators; self.max_depth=max_depth
        self.min_samples_split=min_samples_split; self.n_features_try=n_features_try; self.seed=seed
    def fit(self, X, y):
        rng = np.random.RandomState(self.seed)
        n = len(y)
        self.trees = []
        for i in range(self.n_estimators):
            boot_idx = rng.randint(0, n, size=n)
            Xb, yb = X[boot_idx], y[boot_idx]
            tree = DecisionTreeClassifierNP(max_depth=self.max_depth,
                                             min_samples_split=self.min_samples_split,
                                             n_features_try=self.n_features_try,
                                             seed=self.seed+i)
            tree.fit(Xb, yb)
            self.trees.append(tree)
        return self
    def predict_proba(self, X):
        preds = np.array([t.predict_proba(X) for t in self.trees])
        return preds.mean(axis=0)
    def feature_importances(self, n_features):
        # count split usage weighted by gain proxy (simple frequency-based importance)
        counts = np.zeros(n_features)
        def walk(node):
            if node.value is None:
                counts[node.feature] += 1
                walk(node.left); walk(node.right)
        for t in self.trees:
            walk(t.root)
        if counts.sum() == 0: return counts
        return counts / counts.sum()


class GradientBoostingClassifierNP:
    """Reference stand-in for XGBoost: additive shallow regression trees on
    the negative gradient of log-loss, with shrinkage (learning rate)."""
    def __init__(self, n_estimators=60, max_depth=3, learning_rate=0.15, min_samples_split=20, seed=42):
        self.n_estimators=n_estimators; self.max_depth=max_depth
        self.learning_rate=learning_rate; self.min_samples_split=min_samples_split; self.seed=seed
    def fit(self, X, y):
        n = len(y)
        self.init_logit = np.log(y.mean()/(1-y.mean()+1e-12) + 1e-12)
        F = np.full(n, self.init_logit)
        self.trees = []
        for i in range(self.n_estimators):
            p = 1/(1+np.exp(-F))
            grad = y - p  # negative gradient of log-loss for this direction
            tree = DecisionTreeRegressorNP(max_depth=self.max_depth,
                                            min_samples_split=self.min_samples_split,
                                            seed=self.seed+i)
            tree.fit(X, grad)
            update = tree.predict(X)
            F += self.learning_rate * update
            self.trees.append(tree)
        return self
    def predict_raw(self, X):
        F = np.full(X.shape[0], self.init_logit)
        for tree in self.trees:
            F += self.learning_rate * tree.predict(X)
        return F
    def predict_proba(self, X):
        F = self.predict_raw(X)
        return 1/(1+np.exp(-F))
    def feature_importances(self, n_features):
        counts = np.zeros(n_features)
        def walk(node):
            if node.value is None:
                counts[node.feature] += 1
                walk(node.left); walk(node.right)
        for t in self.trees:
            walk(t.root)
        if counts.sum() == 0: return counts
        return counts / counts.sum()
