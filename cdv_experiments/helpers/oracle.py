"""
Oracle estimator selection via inner cross-validation.

Uses TRUE CATE as the selection target (oracle in the sense that true CATE is
unavailable in deployment, but known in our evaluation setup).

Selection happens entirely within the outer training data.
The selected learner is then refitted on the FULL outer training set before
test evaluation.
"""
import numpy as np
from copy import deepcopy
from sklearn.model_selection import KFold


def inner_cv_oracle_select(
    X_train: np.ndarray,
    t_train: np.ndarray,
    y_train: np.ndarray,
    true_cate_train: np.ndarray,
    estimator_grid: dict,
    n_folds: int,
    seed: int,
) -> tuple:
    """
    Select the best learner using inner k-fold CV with true CATE as target.

    When n_folds == 1 (default), uses leave-nothing-out: fits and evaluates
    on the same training data (biased but computationally feasible).

    Parameters
    ----------
    X_train, t_train, y_train : arrays
        Outer training data.
    true_cate_train : array
        Oracle true CATE for the outer training cases.
    estimator_grid : dict
        {learner_name: estimator_instance}
    n_folds : int
        Number of inner CV folds. 1 = train-on-all.
    seed : int
        Seed for KFold shuffle.

    Returns
    -------
    selected_learner : str
        Name of the best-performing learner.
    cv_scores : dict
        {learner_name: mean_inner_cate_mse}
    """
    cv_scores: dict = {}

    if n_folds <= 1:
        # In-sample evaluation (biased selection, but fast)
        for name, est in estimator_grid.items():
            try:
                e = deepcopy(est)
                e.fit(X_train, t_train, y_train)
                y0 = e.predict_outcome(X_train, np.zeros_like(t_train)).flatten()
                y1 = e.predict_outcome(X_train, np.ones_like(t_train)).flatten()
                ite_pred = y1 - y0
                mse = float(np.nanmean((ite_pred - true_cate_train) ** 2))
            except Exception:
                mse = float("inf")
            cv_scores[name] = mse
    else:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        fold_scores: dict = {name: [] for name in estimator_grid}

        for train_idx, val_idx in kf.split(X_train):
            Xf, tf, yf = X_train[train_idx], t_train[train_idx], y_train[train_idx]
            Xv, tv = X_train[val_idx], t_train[val_idx]
            cate_v = true_cate_train[val_idx]

            for name, est in estimator_grid.items():
                try:
                    e = deepcopy(est)
                    e.fit(Xf, tf, yf)
                    y0 = e.predict_outcome(Xv, np.zeros_like(tv)).flatten()
                    y1 = e.predict_outcome(Xv, np.ones_like(tv)).flatten()
                    ite_pred = y1 - y0
                    mse = float(np.nanmean((ite_pred - cate_v) ** 2))
                except Exception:
                    mse = float("inf")
                fold_scores[name].append(mse)

        cv_scores = {name: float(np.mean(sc)) for name, sc in fold_scores.items()}

    # Select learner with lowest inner-CV CATE MSE
    selected = min(cv_scores, key=lambda k: cv_scores[k])
    return selected, cv_scores


def fit_oracle_and_predict(
    X_train: np.ndarray,
    t_train: np.ndarray,
    y_train: np.ndarray,
    true_cate_train: np.ndarray,
    X_test: np.ndarray,
    t_test: np.ndarray,
    estimator_grid: dict,
    n_folds: int,
    seed: int,
) -> dict:
    """
    Full oracle pipeline: select learner via inner CV, refit on full training data,
    predict on test.

    Returns dict with keys:
        selected_learner, cv_scores, ite_pred_test
    """
    selected, cv_scores = inner_cv_oracle_select(
        X_train, t_train, y_train, true_cate_train, estimator_grid, n_folds, seed
    )

    # Refit selected learner on full outer training data
    est = deepcopy(estimator_grid[selected])
    try:
        est.fit(X_train, t_train, y_train)
        y0 = est.predict_outcome(X_test, np.zeros_like(t_test)).flatten()
        y1 = est.predict_outcome(X_test, np.ones_like(t_test)).flatten()
        ite_pred = y1 - y0
    except Exception as e:
        ite_pred = np.full(len(X_test), np.nan)

    return {
        "selected_learner": selected,
        "cv_scores": cv_scores,
        "ite_pred_test": ite_pred,
    }
