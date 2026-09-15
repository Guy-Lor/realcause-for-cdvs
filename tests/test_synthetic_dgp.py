import numpy as np

from cdv_utils.synthetic_dgp import (
    MISSING_VALUE,
    _propensity_for_subgroup,
    generate_counterfactuals_for_fixed_features,
    generate_synthetic_dataset,
)


def test_sg0_structural_subgroups_share_observed_feature_set():
    df = generate_synthetic_dataset(n=5000, alpha=1.0, seed=42)
    sg0 = df[df["subgroup"] == 0]

    assert set(sg0["structural_subgroup"].unique()) == {"SG0a", "SG0b"}

    for _, row in sg0.iterrows():
        assert row["V"] > MISSING_VALUE
        assert row["Z1"] > MISSING_VALUE
        assert row["E"] == MISSING_VALUE
        assert row["Z2"] == MISSING_VALUE


def test_sg0_structural_subgroups_share_decision_mechanism_given_history():
    x1 = np.array([1.0, 2.0, 3.0])
    x2 = np.array([0.0, 1.0, 0.0])
    features = {
        "V": np.array([1.5, 2.5, 4.0]),
        "Z1": np.array([2.0, 3.0, 1.0]),
    }

    p_sg0a = _propensity_for_subgroup(0, x1, x2, features, structural_subgroup="SG0a")
    p_sg0b = _propensity_for_subgroup(0, x1, x2, features, structural_subgroup="SG0b")

    np.testing.assert_allclose(p_sg0a, p_sg0b)


def test_counterfactual_regeneration_preserves_sg0_structural_metadata():
    df = generate_synthetic_dataset(n=1000, alpha=1.0, seed=42)
    feature_cols = ["X1", "X2", "V", "E", "Z1", "Z2", "subgroup", "structural_subgroup"]

    regenerated = generate_counterfactuals_for_fixed_features(
        df[feature_cols],
        alpha=0.0,
        seed=7,
    )

    np.testing.assert_array_equal(
        regenerated["structural_subgroup"].values,
        df["structural_subgroup"].values,
    )
    np.testing.assert_allclose(regenerated["ite"].values, 5.0)
