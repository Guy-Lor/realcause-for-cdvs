# Configuration Guide — Revised CDV Experiment

This document explains every configurable parameter in `sepsis/config.py` and
`synthetic/config.py`.  For each parameter the guide shows the default value,
valid range, and the effect of increasing or decreasing it.

---

## Experiment Scale

### `N_OUTER_SEEDS` (default: 20)

The number of independent experimental replications.  Each outer seed generates
a new stochastic realisation of the data (new T/Y draw for Sepsis, new DGP
sample for synthetic) and a new train/test split.  All methods and learners
within one outer seed share the exact same data.

| Value | Effect |
|-------|--------|
| 10    | Minimum acceptable; wide confidence intervals |
| 15    | Acceptable compromise |
| **20** | **Target** |
| 50+   | Tighter CIs; substantially longer runtime |

---

## Train / Test Split (Sepsis only)

### `TRAIN_PROP` / `TEST_PROP` (default: 0.70 / 0.30)

Fraction of all Sepsis cases used for training and testing.  Must sum to 1.

| Scenario | TRAIN_PROP | TEST_PROP | Effect |
|----------|-----------|-----------|--------|
| More training data | 0.80 | 0.20 | Larger CDVs; fewer test cases for evaluation |
| More test data     | 0.60 | 0.40 | Smaller CDVs; more stable test estimates |
| **Default**        | **0.70** | **0.30** | Balanced |

---

## CDV Discovery

### `CDV_COVERAGE_THRESHOLD` (default: 0.80)

The greedy selection algorithm adds feature patterns (sorted by frequency) one
by one until cumulative training-data coverage first reaches or exceeds this
threshold.  All remaining patterns are collapsed into the **OTHER** bucket.

| Value | Effect |
|-------|--------|
| 0.60  | Only 1–2 large CDVs retained; large OTHER bucket (≥40 %) |
| **0.80** | **Typically 2–4 CDVs retained; ~20 % in OTHER** |
| 0.95  | Many CDVs retained; tiny OTHER bucket; risk of very small CDVs |
| 1.00  | All patterns become CDVs; no OTHER bucket |

*Applies to training data only.*

### `CDV_N_MIN` (default: 20)

Minimum number of training cases in a CDV for it to be retained.  CDVs with
fewer cases are sent to OTHER.

| Value | Effect |
|-------|--------|
| 5     | Very permissive; tiny CDVs may produce unstable estimates |
| **20** | **Recommended minimum for stable RF fitting** |
| 50    | Stricter; more CDVs fall to OTHER in small datasets |

### `CDV_MIN_ARM_SIZE` (default: 2)

Minimum number of treated cases **and** minimum number of control cases
required for a CDV to be retained.  Prevents near-deterministic treatment
assignment.

| Value | Effect |
|-------|--------|
| 1     | Allows CDVs where almost everyone is treated or control |
| **2**  | **Current behaviour; both arms must have ≥ 2 cases** |
| 5     | Stricter positivity requirement |

---

## Overlap Check (Option C)

The overlap check has two components (**A** and **B**).  A CDV must pass
**both** to be retained.

### `OVERLAP LO` / `OVERLAP HI` (default: 0.10 / 0.90)

Defines the overlap interval $[\text{lo}, \text{hi}]$.

**Option A** (group-level): the mean treatment rate within the CDV must be
inside $[\text{OVERLAP LO}, \text{OVERLAP HI}]$.

**Option B** (individual-level): the fraction of CDV training cases whose
estimated propensity $\hat{p}(T=1|W)$ falls inside $[\text{lo}, \text{hi}]$
must be ≥ `OVERLAP_MIN_FRACTION`.

| Value | Effect |
|-------|--------|
| lo=0.05, hi=0.95 | Very permissive; almost all CDVs pass |
| **lo=0.10, hi=0.90** | **Standard epidemiological threshold** |
| lo=0.15, hi=0.85 | Strict; more CDVs fail and go to OTHER |

### `OVERLAP_MIN_FRACTION` (default: 0.70)

Minimum fraction of a CDV's training cases that must individually have
$\hat{p}$ inside $[\text{OVERLAP LO}, \text{OVERLAP HI}]$.

| Value | Effect |
|-------|--------|
| 0.50  | Permissive; CDVs with moderate overlap pass |
| **0.70** | **70 % of cases must be in the overlap zone** |
| 0.90  | Very strict; most CDVs will fail unless overlap is excellent |

---

## Estimators

### `RF_N_ESTIMATORS` (default: 100)

Number of trees in every RandomForest model (outcome, propensity, CATE).

| Value | Effect |
|-------|--------|
| 50    | Faster; current original experiment setting |
| **100** | **More stable estimates; recommended** |
| 200   | Slower but more stable; diminishing returns |

### `DR_FINAL_MODEL` (default: `"rf"`)

The final-stage model inside the DR Learner.

| Value    | Effect |
|----------|--------|
| `"rf"`    | Fully non-parametric CATE; **primary for paper** |
| `"linear"` | Linear final stage; more interpretable; faster |

### `DR_CV` (default: 1)

Number of cross-fitting folds inside the DR Learner itself.

| Value | Effect |
|-------|--------|
| **1**  | **No cross-fitting; current behaviour; faster** |
| 5      | Proper cross-fitting; recommended for final paper results |

---

## Matched Random Partitions

### `N_RANDOM_PERMUTATIONS` (default: 15)

Number of random-partition permutations generated per outer seed.  These are
**not** independent experimental replications.  Confidence intervals are always
based on the outer seeds.

| Value | Effect |
|-------|--------|
| 5     | Fast; noisier average estimate |
| **15** | **Reasonable balance** |
| 30    | Smoother average; ~2× runtime for this method |

---

## Oracle Selection

### `N_ORACLE_CV_FOLDS` (default: 1)

Inner-CV folds used to select the best learner within the outer training data,
using true CATE as the selection target.

| Value | Effect |
|-------|--------|
| **1**  | **In-sample evaluation; biased selection; fast** |
| 3      | 3-fold CV; less biased |
| 5      | Standard k-fold; recommended for robust oracle |

---

## Confidence Intervals

### `CI_METHOD` (default: `"t"`)

Method for computing paired confidence intervals across outer seeds.

| Value       | Effect |
|-------------|--------|
| `"t"`        | Paired t-based CI; assumes normality of seed-level deltas |
| `"bootstrap"` | Bootstrap CI; non-parametric; 10 000 resamples |

---

## Synthetic DGP (synthetic/config.py only)

### `ALPHA_VALUES` (default: `[0.0, 0.25, 0.50, 0.75, 1.0]`)

The heterogeneity sweep values.  At $\alpha = 0$ all sub-groups have the same
treatment effect (≈ 5).  At $\alpha = 1$ sub-groups have strongly different,
nonlinear CATEs.

### `N_TRAIN` / `N_TEST` (default: 7000 / 3000)

Fresh datasets are generated per outer seed × alpha.  Increasing N_TRAIN gives
better-fitted CDV models but longer runtime.

| N_TRAIN | Effect |
|---------|--------|
| 2000    | Small; fast debugging runs |
| **7000** | **Paper setting** |
| 20000   | High-powered but slow |
