import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest

class EnsembleBaselines:
    """
    Trains RF + GB classifier on flattened windows + static,
    and an IsolationForest score for anomaly-ish behaviour.
    """
    def __init__(self, seed=42):
        self.rf = RandomForestClassifier(n_estimators=200, random_state=seed, class_weight="balanced_subsample")
        self.gb = GradientBoostingClassifier(random_state=seed)
        self.iforest = IsolationForest(contamination=0.07, random_state=seed)

    def _flat(self, X_ts, X_static):
        n, w, c = X_ts.shape
        return np.hstack([X_ts.reshape(n, w*c), X_static])

    def fit(self, X_ts, X_static, y):
        X = self._flat(X_ts, X_static)
        self.rf.fit(X, y)
        self.gb.fit(X, y)
        self.iforest.fit(X)  # unsupervised drift/anomaly-ish
        return self

    def predict_proba(self, X_ts, X_static):
        X = self._flat(X_ts, X_static)
        p1 = self.rf.predict_proba(X)[:, 1]
        p2 = self.gb.predict_proba(X)[:, 1]
        # map IF decision_function to 0..1 “risk-like” score
        s = self.iforest.decision_function(X)
        s = (s - s.min()) / (s.max() - s.min() + 1e-9)
        # ensemble average (classifiers dominate, IF acts as weak signal)
        p = 0.45*p1 + 0.45*p2 + 0.10*(1.0 - s)
        return p


class RFGBEnsemble:
    """
    Simple, strong baseline:
      - RandomForest (handles nonlinearity, robust)
      - GradientBoosting (strong tabular baseline)
    Ensemble = average of predicted probabilities.
    """
    def __init__(self, seed=42):
        self.rf = RandomForestClassifier(
            n_estimators=400,
            random_state=seed,
            class_weight="balanced_subsample",
            n_jobs=-1
        )
        self.gb = GradientBoostingClassifier(random_state=seed)

    def fit(self, X, y):
        self.rf.fit(X, y)
        self.gb.fit(X, y)
        return self

    def predict_proba(self, X) -> np.ndarray:
        p1 = self.rf.predict_proba(X)[:, 1]
        p2 = self.gb.predict_proba(X)[:, 1]
        return 0.5 * p1 + 0.5 * p2