import numpy as np
import pandas as pd

from cdv_utils.causal_modeling import select_best_model_per_variant
import cdv_utils.experiment_runner as experiment_runner
from cdv_utils.experiment_runner import _select_best_estimator_from_scores


def test_global_selection_separates_ate_mse_from_pehe():
    scores = {
        "constant_ate": {
            "ate_bias": 0.0,
            "ate_mse": 0.0,
            "mse": 25.0,
            "pehe": 5.0,
            "r2": 0.0,
        },
        "heterogeneous_cate": {
            "ate_bias": 0.5,
            "ate_mse": 0.25,
            "mse": 0.5,
            "pehe": np.sqrt(0.5),
            "r2": 0.98,
        },
    }

    best_ate = _select_best_estimator_from_scores(
        scores, selection_metric="ate_mse", use_r2_threshold=False
    )
    best_cate = _select_best_estimator_from_scores(
        scores, selection_metric="pehe", use_r2_threshold=False
    )

    assert best_ate["estimator"] == "constant_ate"
    assert best_ate["selection_metric"] == "ate_mse"
    assert best_cate["estimator"] == "heterogeneous_cate"
    assert best_cate["selection_metric"] == "pehe"


def test_variant_selection_uses_corrected_validation_effects():
    validation_predictions = {
        1: {
            "constant_ate": pd.DataFrame({"ite_pred": [5.0, 5.0]}),
            "heterogeneous_cate": pd.DataFrame({"ite_pred": [0.0, 9.0]}),
        }
    }
    legacy_validation = {
        1: pd.DataFrame({"ite": [1000.0, -1000.0]})
    }
    corrected_cate = {1: np.array([0.0, 10.0])}

    best_ate = select_best_model_per_variant(
        validation_predictions,
        legacy_validation,
        selection_metric="ate_mse",
        validation_effects_by_variant=corrected_cate,
        use_r2_threshold=False,
    )
    best_cate = select_best_model_per_variant(
        validation_predictions,
        legacy_validation,
        selection_metric="pehe",
        validation_effects_by_variant=corrected_cate,
        use_r2_threshold=False,
    )

    assert best_ate[1]["estimator"] == "constant_ate"
    assert best_cate[1]["estimator"] == "heterogeneous_cate"
    assert best_cate[1]["pehe"] == np.sqrt(0.5)


def test_multi_seed_runner_computes_validation_cate_once(monkeypatch, tmp_path):
    quadrature_calls = []
    single_seed_calls = []

    def fake_cate(model, w, n_quantiles, batch_rows):
        quadrature_calls.append((w.copy(), n_quantiles, batch_rows))
        return 2.0 * w[:, 0]

    def fake_single_seed(*args):
        single_seed_calls.append(args[-1])
        return {"seed": args[1]}

    monkeypatch.setattr(
        experiment_runner, "estimate_sigmoid_flow_cate", fake_cate
    )
    monkeypatch.setattr(
        experiment_runner, "run_single_seed_experiment", fake_single_seed
    )

    global_validation = {
        1: pd.DataFrame({"x": [1.0, 3.0], "ite": [99.0, -99.0]})
    }
    result_path = tmp_path / "semi_synthetic.pkl"

    results = experiment_runner.run_multi_seed_experiment(
        best_model=object(),
        experiment_seeds=[10, 11],
        w_cols=["x"],
        top_variants=[],
        k=1,
        training_variant_patterns={},
        test_variant_dataframes={},
        val_variant_dataframes={},
        global_test_variant_dataframes={},
        global_val_variant_dataframes=global_validation,
        results_save_path=str(result_path),
        cate_n_quantiles=32,
        cate_batch_rows=4,
    )

    assert len(quadrature_calls) == 1
    assert quadrature_calls[0][1:] == (32, 4)
    assert len(single_seed_calls) == 2
    assert all(np.array_equal(call[1], np.array([2.0, 6.0])) for call in single_seed_calls)
    assert sorted(results) == [10, 11]
