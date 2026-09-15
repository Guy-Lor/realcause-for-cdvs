"""
Deterministic sub-seed derivation for the revised CDV experiment.

All randomness traces back to one outer_seed integer via numpy SeedSequence.
No two components share the same child seed.
"""
import numpy as np

# Named positions in the child-seed array
_CHILD_NAMES = [
    "draw_train",   # RealCause T/Y sampling (sepsis) or DGP train generation (synthetic)
    "draw_test",    # DGP test generation (synthetic only; unused in sepsis)
    "split",        # stratified train/test split (sepsis only)
    "model",        # estimator RF random_state
    "oracle_cv",    # inner-CV folds for oracle learner selection
]
_N_PERM_SEEDS = 50  # pre-allocate permutation seeds (enough for up to 50 permutations)
_N_CDV_BOOTSTRAP_SEEDS = 50  # bootstrap seeds for CDV_SEPARATE ensemble bagging


def derive_seeds(outer_seed: int) -> dict:
    """
    Return a dict of named child seeds derived deterministically from outer_seed.

    Keys:
        draw_train, draw_test, split, model, oracle_cv  – single int each
        permutations – list of N_PERM_SEEDS ints for random-partition permutations
        cdv_bootstrap – list of N_CDV_BOOTSTRAP_SEEDS ints for CDV ensemble bagging
    """
    ss = np.random.SeedSequence(int(outer_seed))
    children = ss.spawn(len(_CHILD_NAMES) + _N_PERM_SEEDS + _N_CDV_BOOTSTRAP_SEEDS)
    seeds: dict = {}
    for i, name in enumerate(_CHILD_NAMES):
        seeds[name] = int(children[i].generate_state(1)[0])
    seeds["permutations"] = [
        int(children[len(_CHILD_NAMES) + i].generate_state(1)[0])
        for i in range(_N_PERM_SEEDS)
    ]
    seeds["cdv_bootstrap"] = [
        int(children[len(_CHILD_NAMES) + _N_PERM_SEEDS + i].generate_state(1)[0])
        for i in range(_N_CDV_BOOTSTRAP_SEEDS)
    ]
    return seeds
