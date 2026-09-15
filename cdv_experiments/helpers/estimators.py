"""
Learner grid construction for the revised CDV experiment.

Builds the full estimator dictionary from config.
Provides a special LinearSLearnerCDV for the GLOBAL_MISSINGNESS_CDV + S_Linear case.
"""
import sys
import os
import numpy as np
from copy import deepcopy

# Ensure project root is on path so existing causal_estimators are importable
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression

from causal_estimators.metalearners import SLearner, TLearner, XLearner
from causal_estimators.doubly_robust_estimator import DoublyRobustLearner
from causal_estimators.double_ml import DoubleML


# ============================================================
# STANDARD LEARNER GRID
# ============================================================

def build_estimator_grid(config: dict, seed: int) -> dict:
    """
    Build the full learner grid from config parameters.

    Returns a dict:
        {learner_name: estimator_instance}

    Learners: DR_RF, S_RF, S_Linear, T_RF, X_RF, Double_ML
    """
    n_est = int(config.get("RF_N_ESTIMATORS", 100))
    dr_cv = int(config.get("DR_CV", 1))
    final_model_type = config.get("DR_FINAL_MODEL", "rf").lower()

    rf_reg = RandomForestRegressor(n_estimators=n_est, random_state=seed, n_jobs=-1)
    rf_clf = RandomForestClassifier(n_estimators=n_est, random_state=seed, n_jobs=-1)

    if final_model_type == "rf":
        final_model = RandomForestRegressor(n_estimators=n_est, random_state=seed, n_jobs=-1)
    else:
        final_model = LinearRegression()

    estimators = {
        "DR_RF": DoublyRobustLearner(
            outcome_model=deepcopy(rf_reg),
            prop_score_model=deepcopy(rf_clf),
            final_model=deepcopy(final_model),
            trim_eps=0.1,
            random_state=seed,
            cv=dr_cv,
        ),
        "S_RF": SLearner(outcome_model=deepcopy(rf_reg)),
        "S_Linear": SLearner(outcome_model=LinearRegression()),
        "T_RF": TLearner(outcome_models=deepcopy(rf_reg)),
        "X_RF": XLearner(
            outcome_models=deepcopy(rf_reg),
            cate_models=deepcopy(rf_reg),
            prop_score_model=deepcopy(rf_clf),
        ),
        "Double_ML": DoubleML(
            outcome_model=deepcopy(rf_reg),
            prop_score_model=deepcopy(rf_clf),
            final_model=LinearRegression(),
            discrete_treatment=True,
            random_state=seed,
        ),
    }
    return estimators


# ============================================================
# SPECIAL ESTIMATOR: Linear S-Learner with CDV interactions
# ============================================================

class LinearSLearnerCDV:
    """
    Linear S-learner that explicitly includes T × is_CDV_k interaction terms.

    Used only for GLOBAL_MISSINGNESS_CDV + S_Linear.
    The feature matrix passed to fit/predict_outcome must NOT include T or
    interaction terms — this class adds them internally.

    Parameters
    ----------
    n_cdv_cols : int
        Number of CDV one-hot columns at the END of the feature matrix.
        Must be set before calling fit().
    """

    def __init__(self, n_cdv_cols: int = None):
        self._model = LinearRegression()
        self.n_cdv_cols = n_cdv_cols

    def fit(self, X: np.ndarray, t: np.ndarray, y: np.ndarray):
        if self.n_cdv_cols is None:
            raise ValueError("LinearSLearnerCDV.n_cdv_cols must be set before fit().")
        t_col = t.flatten().reshape(-1, 1)
        cdv_part = X[:, -self.n_cdv_cols:]
        interactions = cdv_part * t_col
        X_aug = np.concatenate([X, interactions, t_col], axis=1)
        self._model.fit(X_aug, y.flatten())
        return self

    def predict_outcome(self, X: np.ndarray, t) -> np.ndarray:
        """Predict E[Y | X, T=t] correctly updating interaction terms."""
        n = len(X)
        t_scalar = float(t) if np.isscalar(t) else None
        if t_scalar is not None:
            t_arr = np.full((n, 1), t_scalar)
        else:
            t_arr = np.asarray(t).flatten().reshape(-1, 1)
        cdv_part = X[:, -self.n_cdv_cols:]
        interactions = cdv_part * t_arr
        X_aug = np.concatenate([X, interactions, t_arr], axis=1)
        return self._model.predict(X_aug)
