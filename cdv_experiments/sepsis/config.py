# ============================================================
# CONFIG: Revised Sepsis / RealCause Experiment
# See CONFIG_GUIDE.md for full parameter explanations.
# ============================================================

# --- PATHS ---
# Raw event log (source of truth; never modified)
RAW_XES_PATH = "datasets/Sepsis Cases - Event Log.xes.gz"

# Processed dataset produced by 00_data_preparation.ipynb
DATASET_PATH = "cdv_experiments/sepsis/artifacts/sepsis_cases.csv"

# Trained TarNet checkpoint produced by 01_realcause_training.ipynb
REALCAUSE_CHECKPOINT = "cdv_experiments/sepsis/artifacts/realcause_model/medium_seed_420/model.pt"

# Checkpoint file: experiment results saved here after every outer seed
CHECKPOINT_PATH = "cdv_experiments/sepsis/artifacts/results_checkpoint.pkl"

# Artifact directory
ARTIFACTS_DIR = "cdv_experiments/sepsis/artifacts"

# Plots directory
PLOTS_DIR = "cdv_experiments/sepsis/artifacts/plots"

# --- EXPERIMENT SCALE ---
# Number of independent outer-seed replications.
# Target: 20. Minimum acceptable: 10.
N_OUTER_SEEDS = 20

# Outer seeds to run.  Modify the range to add/remove seeds.
OUTER_SEEDS = list(range(N_OUTER_SEEDS))

# --- TRAIN / TEST SPLIT ---
# Fraction of all cases used for training (the rest go to test).
# Must satisfy TRAIN_PROP + TEST_PROP = 1.0.
TRAIN_PROP = 0.70
TEST_PROP  = 0.30

# Stratification column for the split.  "t" stratifies by generated treatment,
# ensuring similar treatment rates in both folds.
STRATIFY_BY = "t"

# --- CDV DISCOVERY (training data only) ---
# Greedy cumulative-coverage threshold.  Add patterns one by one (most frequent
# first) until cumulative coverage of training cases >= this value.
CDV_COVERAGE_THRESHOLD = 0.90

# Minimum number of training cases required for a CDV to be retained.
# CDVs below this size are sent to OTHER.
CDV_N_MIN = 50

# Minimum number of cases in each treatment arm (t=0, t=1) for a CDV
# to be retained.  Mirrors the current two-case-per-arm check.
CDV_MIN_ARM_SIZE = 2

# --- OVERLAP CHECK (Option C: both A and B required) ---
# Lower bound of the overlap / positivity interval.
OVERLAP_LO = 0.10

# Upper bound of the overlap / positivity interval.
OVERLAP_HI = 0.90

# Minimum fraction of CDV training cases whose propensity score p̂(T=1|W)
# must fall inside [OVERLAP_LO, OVERLAP_HI] for the CDV to be retained.
OVERLAP_MIN_FRACTION = 0.20

# --- ESTIMATORS ---
# Number of trees in each RandomForest model.
RF_N_ESTIMATORS = 100

# Final model for the DR Learner.
#   "rf"     → RandomForestRegressor (primary, fully non-parametric)
#   "linear" → LinearRegression (more interpretable, faster)
DR_FINAL_MODEL = "linear"

# Cross-fitting folds inside the DR Learner.
#   1 → no cross-fitting (current behaviour, faster)
#   4 → standard cross-fitting
DR_CV = 3

# --- MATCHED RANDOM PARTITIONS ---
# Number of random-partition permutations run per outer seed.
# These are NOT independent replications; CIs are based on outer seeds.
N_RANDOM_PERMUTATIONS = 15

# --- CDV BOOTSTRAP ENSEMBLE ---
# Number of bootstrap resamples used when fitting each CDV_SEPARATE model.
# Matches MATCHED_RANDOM_PARTITIONS' variance reduction.  1 = no ensembling.
N_CDV_BOOTSTRAP = 1

# --- ORACLE SELECTION ---
# Inner-CV folds used to select the oracle estimator within the outer training data.
#   1 → single in-sample evaluation (biased but fast)
#   5 → k-fold cross-validation (preferred for a rigorous oracle)
N_ORACLE_CV_FOLDS = 2

# --- SENTINEL ---
# Value used to represent structurally absent features.
# In the Sepsis dataset: NaN in the CSV becomes -1 via load_sepsis().
SENTINEL_VALUE = -1

# --- POSITIVITY ENFORCEMENT ---
# If > 0, propensity scores are clipped to [POSITIVITY_CLIP, 1 - POSITIVITY_CLIP]
# before sampling T each outer seed. This enforces overlap in the synthetic DGP
# so that CDV subgroups reliably pass the retention check.
# 0.0 = off (use raw model propensities). Default: 0.25.
POSITIVITY_CLIP = 0.0

# --- CONFIDENCE INTERVALS ---
# Method for paired confidence intervals across outer seeds.
#   "t"         → paired t-based CI (parametric, assumes normality of deltas)
#   "bootstrap" → bootstrap CI (non-parametric, 10 000 resamples)
CI_METHOD = "bootstrap"

# Sidedness of every paired CI/test (all CIs/tests in this project are paired
# across outer seeds; see helpers.metrics.paired_ci).
#   "one-sided" → tests/bounds H1: CDV_SEPARATE (or method) is strictly BETTER
#                 than the comparator (Δ < 0 for MSE metrics, Δ > 0 for rank
#                 correlation metrics). Matches the paper's directional claim.
#   "two-sided" → tests/bounds H1: Δ ≠ 0 (any difference, either direction).
CI_SIDED = "two-sided"

# --- PRIMARY LEARNER ---
# Learner used for primary (paper-facing) results tables and figures.
# Must match a key in the estimator grid: DR_RF, S_RF, S_Linear, T_RF, X_RF.
PRIMARY_LEARNER = "DR_RF"
