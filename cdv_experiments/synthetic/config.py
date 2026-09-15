# ============================================================
# CONFIG: Revised Synthetic DGP Experiment
# See CONFIG_GUIDE.md for full parameter explanations.
# ============================================================

# --- PATHS ---
ARTIFACTS_DIR       = "cdv_experiments/synthetic/artifacts"
PLOTS_DIR           = "cdv_experiments/synthetic/artifacts/plots"

# Checkpoint template: one file per alpha value.
# The string '{alpha:.2f}' is replaced with the actual alpha at runtime.
CHECKPOINT_PATH_TEMPLATE = "cdv_experiments/synthetic/artifacts/results_alpha_{alpha:.2f}.pkl"

# --- DGP ---
# Heterogeneity sweep values.  At alpha=0 true ITEs are constant (=5).
# At alpha=1 sub-groups have strongly different, nonlinear CATEs.
ALPHA_VALUES = [0.0, 0.25, 0.50, 0.75, 1.0]

# Training samples generated per seed × alpha combination.
N_TRAIN = 7000

# Test samples generated per seed × alpha combination.
# A fresh test set is drawn for every outer seed (unlike the original experiment).
N_TEST = 3000

# Sentinel value for absent features in the synthetic DGP.
SENTINEL_VALUE = -100.0

# --- EXPERIMENT SCALE ---
N_OUTER_SEEDS = 20
OUTER_SEEDS   = list(range(N_OUTER_SEEDS))

# --- CDV DISCOVERY (training data only) ---
CDV_COVERAGE_THRESHOLD = 0.80
CDV_N_MIN              = 100
CDV_MIN_ARM_SIZE       = 10

# --- OVERLAP CHECK ---
OVERLAP_LO           = 0.10
OVERLAP_HI           = 0.90
OVERLAP_MIN_FRACTION = 0.70

# --- ESTIMATORS ---
RF_N_ESTIMATORS = 100
DR_FINAL_MODEL  = "linear"
DR_CV           = 4

# --- MATCHED RANDOM PARTITIONS ---
N_RANDOM_PERMUTATIONS = 15

# --- CDV BOOTSTRAP ENSEMBLE ---
N_CDV_BOOTSTRAP = 1

# --- ORACLE SELECTION ---
N_ORACLE_CV_FOLDS = 2

# --- CONFIDENCE INTERVALS ---
CI_METHOD = "bootstrap"

# Sidedness of every paired CI/test (all CIs/tests in this project are paired
# across outer seeds; see helpers.metrics.paired_ci).
#   "one-sided" → tests/bounds H1: CDV_SEPARATE (or method) is strictly BETTER
#                 than the comparator (Δ < 0 for MSE metrics, Δ > 0 for rank
#                 correlation metrics). Matches the paper's directional claim.
#   "two-sided" → tests/bounds H1: Δ ≠ 0 (any difference, either direction).
CI_SIDED = "two-sided"

# --- PRIMARY LEARNER ---
PRIMARY_LEARNER = "DR_RF"
