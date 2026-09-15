# Synthetic CDV Experiment and Evaluation

This document explains the experiment implemented in
`05_synthetic_experiment.ipynb` and the evaluation implemented in
`06_synthetic_analysis.ipynb`. It is written for a reader who has not seen the
notebooks or the synthetic data generating process before.

## Purpose

The experiment tests whether causal decision variant (CDV) modeling improves
treatment-effect estimation when different observed process paths imply
different causal response surfaces.

The comparison is between two modeling strategies:

1. **Global modeling:** fit one causal model on the full population.
2. **CDV modeling:** split the population by observed feature-availability
   patterns, then fit separate causal models for the main variants.

The synthetic setup is controlled, so the true potential outcomes `y0`, `y1`
and true individual treatment effect `ite = y1 - y0` are known for every case.
This allows direct evaluation of ATE and CATE/ITE accuracy.

The key experimental knob is `alpha`, the heterogeneity strength:

- `alpha = 0`: all variants share a constant treatment effect. There is no true
  cross-variant treatment-effect heterogeneity.
- `alpha = 1`: each process path has its own nonlinear treatment-effect
  function.
- Intermediate alpha values interpolate between these cases.

In notebook 05 the configured alpha sweep is:

```text
[0.0, 0.25, 0.5, 0.75, 1.0]
```

## Data Generating Process

The synthetic DGP lives in `cdv_utils/synthetic_dgp.py`. It creates an
emergency-room-like process in which patients follow different pathways before
the treatment decision. The pathway determines which measurements are available
at decision time.

The feature columns are:

| Column | Meaning in the synthetic story | Always observed |
|---|---|---|
| `X1` | Baseline acuity or age-like continuous feature | Yes |
| `X2` | Binary baseline feature | Yes |
| `V` | Extended vital-sign measurement | No |
| `E` | ECG-like measurement | No |
| `Z1` | Lab panel | No |
| `Z2` | Specialty test | No |

Missing features are structural: a value is absent because that process path did
not generate the measurement before treatment. Absence is encoded with the
sentinel value `-100.0`, not with `NaN`. This makes the feature-availability
pattern observable to the global model as well.

The DGP has six observed subgroups:

| Subgroup | Population share | Available features | CDV role |
|---|---:|---|---|
| SG0 | 40% | `X1`, `X2`, `V`, `Z1` | Top modeled variant |
| SG1 | 30% | `X1`, `X2`, `E`, `V` | Top modeled variant |
| SG2 | 17% | `X1`, `X2`, `V`, `E`, `Z2` | Top modeled variant |
| SG3 | 5% | `X1`, `X2`, `Z1`, `Z2` | Others bucket |
| SG4 | 4% | `X1`, `X2`, `E`, `Z1` | Others bucket |
| SG5 | 4% | `X1`, `X2`, `V`, `Z1`, `Z2` | Others bucket |

SG0 is intentionally split into two latent structural mechanisms, `SG0a` and
`SG0b`. Both expose the same observed feature set, `X1`, `X2`, `V`, `Z1`, so an
observed CDV method cannot separate them. They differ in feature-generation,
baseline-outcome, and treatment-effect equations, but they share the same
treatment propensity conditional on the observed history. This preserves the
intended assumption that the latent mechanism is not an additional treatment
selection variable once the observed decision history is fixed.

Treatment is binary (`t` in `{0, 1}`). Potential outcomes are generated as:

```text
y1 = y0 + tau
ite = y1 - y0 = tau
y = y0 if t = 0, otherwise y1
```

The treatment effect has the form:

```text
tau_i = TAU_BASE + alpha * delta_variant(X_i)
TAU_BASE = 5.0
```

The `delta_variant` functions differ by process path. The design deliberately
creates conflicting feature meanings across variants. For example, high `V`
increases treatment benefit in SG0 but can reduce benefit in SG1. A single
global model must learn these path-specific interactions from the sentinel-coded
feature table, while CDV models learn local response surfaces after splitting by
path.

The structural equations were chosen to make the main source of heterogeneity
visible at the observed CDV level rather than hidden inside an unobserved latent
mechanism. SG0 still contains the latent `SG0a`/`SG0b` split because it is useful
for the latent-template story: two real clinical pathways can arrive at the same
observed decision history and therefore cannot be separated by feature
availability alone. However, that latent split is not meant to be the main
reason CDV works in this experiment. The central empirical claim is that
observed process histories can define different causal response surfaces, so
SG0, SG1, and SG2 were made clearly different from each other. In SG0 the
treatment-effect modifier is mainly driven by `V` and `Z1`; in SG1 it is driven
by `E` and `V`; and in SG2 it depends on `Z2` together with nonlinear or
opposite-sign effects of `V` and `E`. This makes the observed path itself
important: the same measurement can have a different clinical meaning depending
on which process produced it.

This is also why some relationships intentionally conflict across variants. A
high value of `V` is beneficial evidence in the SG0 response surface, but it can
have the opposite meaning in SG1, while SG2 uses `V` nonlinearly and combines it
with `E` and `Z2`. A global model that sees a single `V` column must represent
all of these incompatible meanings in one function. Because the global model is
given the same sentinel-coded missingness patterns as CDV, this is still a fair
comparison: CDV is not receiving hidden information. Instead, CDV changes the
statistical problem from one pooled function with implicit `variant x feature`
interactions into several simpler local functions. A flexible global random
forest may learn some of these interactions, but with finite samples and limited
tuning it may not learn them as cleanly; a global linear learner is especially
limited unless explicit interaction terms are added. The experiment is therefore
designed to test the argument that CDV helps by giving the estimator the right
local modeling problem, not by withholding information from the global baseline.

For a reader with real-world data, the practical question is whether their data
behaves like this even though the true structural equations are unknown. The
main diagnostic is to treat feature availability and process history as a
substantive object, not only as a missing-data nuisance. If the common
measurement patterns correspond to recognizable workflows, departments,
protocols, triage paths, or ordering sequences, then CDV-style partitioning may
be relevant. The next check is whether relationships between the same covariates
and outcomes differ across those observed paths: for example, fitted outcome
curves, residual plots, treatment propensities, or estimated treatment effects
look different inside each path, or a feature that is predictive in one path has
weaker, nonlinear, or opposite association in another. In real data the true ITE
is not observed, so this cannot be proven directly from ground truth; instead,
one should look for stable path-specific associations, strong feature-pattern
effects, improved validation performance from path-specific models, and domain
knowledge supporting the idea that the same measurement has different meaning
when it is collected by different processes. If those signals are absent and the
same response surface appears to hold across paths, then a global model with
missingness indicators may be sufficient.

## Why K Equals 3

Notebook 04 creates the feature-pattern elbow chart. Feature patterns are binary
strings in the feature order:

```text
['X1', 'X2', 'V', 'E', 'Z1', 'Z2']
```

A `1` means the feature is present and a `0` means the feature is structurally
absent.

The observed pattern frequencies in the synthetic reference data are:

| Rank | Pattern | Features | Share | Cumulative coverage |
|---:|---|---|---:|---:|
| 1 | `111010` | `X1`, `X2`, `V`, `Z1` | 40.3% | 40.3% |
| 2 | `111100` | `X1`, `X2`, `V`, `E` | 30.2% | 70.6% |
| 3 | `111101` | `X1`, `X2`, `V`, `E`, `Z2` | 16.8% | 87.3% |
| 4 | `110011` | `X1`, `X2`, `Z1`, `Z2` | 4.8% | 92.2% |
| 5 | `111011` | `X1`, `X2`, `V`, `Z1`, `Z2` | 4.0% | 96.1% |
| 6 | `110110` | `X1`, `X2`, `E`, `Z1` | 3.9% | 100.0% |

The elbow is after the third pattern: the top three variants cover about 87% of
the population, and each additional pattern adds only about 4-5 percentage
points. Therefore the experiment chooses:

```text
NUM_TOP_VARIANTS = 3
K = NUM_TOP_VARIANTS + 1 = 4
```

Here, `K = 4` means three explicitly modeled CDV variants plus one "others"
bucket. In the prose it is common to say "choose k = 3" because three top
variants are modeled. In the code, `K = 4` because the others bucket is counted
as an additional variant.

## Population, Splits, and Seeds

Notebook 05 is configured with:

| Quantity | Value |
|---|---:|
| Training samples per seed | 7,000 |
| Fixed test samples | 3,000 |
| Fixed validation samples | 3,000 |
| Planned Monte Carlo seeds | 100 |
| Estimator initialization seed | 420 |
| Test/validation feature seed | 999 |
| R2 model-selection threshold | 0.1 |

The test and validation features are generated once and then kept fixed across
all alpha values and experiment seeds. Only treatment assignment and
counterfactual outcomes are regenerated for each alpha. This design removes
feature-sampling variation from the alpha comparison.

The fixed test set observed in notebook 05 has this variant structure:

| Variant | Pattern | Test rows | Share | Model input columns |
|---:|---|---:|---:|---|
| 1 | `111010` | 1,210 | 40.3% | `X1`, `X2`, `V`, `Z1` |
| 2 | `111100` | 906 | 30.2% | `X1`, `X2`, `V`, `E` |
| 3 | `111101` | 515 | 17.2% | `X1`, `X2`, `V`, `E`, `Z2` |
| 4 | others | 369 | 12.3% | all six feature columns |

The validation split has the same role as a tuning split. It is not used to fit
the final evaluation metric. It is used to choose the best estimator for each
seed and, for CDV, for each variant.

Current result artifacts in `06_synthetic_analysis.ipynb` analyze the completed
saved files in `results/synthetic_alpha_*.pkl`. Those files currently contain
13 seeds for alpha `0.0` and `0.25`, and 12 seeds for alpha `0.5`, `0.75`, and
`1.0`. The notebook configuration describes the intended 100-seed run, but the
analysis reflects the saved partial run. Also note that
`results/synthetic_experiment_config.json` currently records a smaller stale
configuration (`n_train=3000`, `n_test=1000`, `n_val=1000`, `num_seeds=2`);
notebook 06 explicitly overrides alpha values and loads the available alpha
files.

## Models

For every seed, the experiment initializes the same family of causal estimators
for both global and CDV modeling. The estimators are defined in
`cdv_utils/causal_modeling.py`.

| Estimator name | Main components |
|---|---|
| `S-Learner (Linear)` | Single outcome model with `LinearRegression` |
| `S-Learner (RF)` | Single outcome model with `RandomForestRegressor(n_estimators=50)` |
| `T-Learner (RF)` | Separate treatment/control outcome models with random forests |
| `X-Learner (RF)` | Random-forest outcome models, random-forest CATE model, random-forest propensity classifier |
| `DR Learner (EconML)` | Random-forest outcome model, random-forest propensity model, linear final model, `trim_eps=0.1` |
| `Double ML` | Random-forest outcome model, random-forest propensity model, linear final model, discrete treatment |

Random-forest estimators use the configured initialization seed and 50 trees.

Each fitted estimator predicts both counterfactual outcomes:

```text
y0_pred = predicted outcome under control
y1_pred = predicted outcome under treatment
ite_pred = y1_pred - y0_pred
```

These predictions are compared with the known `y0`, `y1`, and `ite`.

## Global Model

The global method fits each estimator once on all training rows pooled together.
The global feature matrix contains all six feature columns:

```text
X1, X2, V, E, Z1, Z2
```

Structurally absent measurements remain encoded as `-100.0`. This is important
for fairness: the global model is not blind to the process path. It can infer
the path from the sentinel-coded missingness pattern. However, because it is one
pooled model, it must learn all variant-specific feature meanings and
interactions inside a single response surface.

For evaluation, the same fitted global estimator is applied to each variant's
test rows. The results are then concatenated into:

- `global_method`: all estimators, all test rows.
- `global_method_best`: the validation-ATE-selected global estimator only.
- `global_method_best_cate`: the validation-ITE-MSE-selected global estimator
  only.

Each `global_method` seed result has 18,000 rows in the current 3,000-row test
setup because there are 3,000 test instances times six estimators.

## CDV Modeling

The CDV method first assigns rows to variants by feature-availability pattern.
For each row, the pattern is computed by checking whether each feature value is
greater than or equal to zero:

```text
present(feature) = feature >= 0
pattern = binary string over ['X1', 'X2', 'V', 'E', 'Z1', 'Z2']
```

Rows matching the top three patterns are assigned to variants 1, 2, and 3. All
other patterns are assigned to variant 4.

The CDV training data are then column-filtered:

- Variants 1-3 keep only the features present in that variant's pattern.
- Variant 4, the others bucket, keeps all six feature columns.

This means the top CDV models do not carry sentinel columns for features that
were structurally unavailable on that path. They estimate a local causal model
using the measurements that actually exist in that process context.

The others bucket is deliberately handled conservatively. It uses the best
global model rather than fitting a separate small local model, because it
combines multiple rare patterns and has limited sample size. The code also falls
back to the global model for any variant that lacks sufficient treatment
variation, although the synthetic DGP is designed to satisfy positivity.

CDV outputs are stored as:

- `variant_method`: all estimators, all test rows.
- `variant_method_best`: validation-ATE-selected estimator per variant.
- `variant_method_best_cate`: validation-ITE-MSE-selected estimator per variant.

The `variant_method` outputs include `used_global_model`, which identifies rows
where the CDV path used the global fallback.

## Experiment Loop in Notebook 05

The notebook performs the following steps:

1. Configure alpha values, sample sizes, seeds, feature columns, and the CDV
   variant count.
2. Generate fixed test and validation feature tables using `TEST_VAL_SEED`.
3. Discover the top three feature patterns from the fixed test features.
4. For each alpha, regenerate treatment assignment and counterfactual outcomes
   for the fixed test and validation features.
5. Run the multi-seed experiment in seed-first order. For each seed and alpha:
   generate a fresh training set, fit global estimators, fit CDV estimators,
   predict on validation data for model selection, predict on test data, and
   save results.
6. Save each alpha's results to:

```text
results/synthetic_alpha_{alpha:.2f}.pkl
```

The seed-first loop is resumable. If an alpha/seed result already exists and has
the required keys, it is skipped. If older results are missing the newer CATE
selection keys, that alpha/seed is rerun.

## Best Model Selection

Best models are selected on the validation set, where the true validation ITE is
known.

For every candidate estimator, the validation metrics are:

```text
ate_bias = abs(mean(ite_pred) - mean(ite_real))
ite_mse = mean((ite_pred - ite_real)^2)
ite_r2 = R2(ite_real, ite_pred)
```

There are two parallel model-selection modes.

### ATE-focused selection

This selection is used for `global_method_best` and `variant_method_best`.

1. Filter candidates to those with validation `R2 > 0.1`.
2. Among the valid candidates, choose the estimator with the smallest
   validation `ate_bias`.
3. If no candidate passes the R2 threshold, choose the estimator with the
   smallest validation `ate_bias` anyway.

For the global method, the validation predictions are combined across all
variants before computing the global score. For CDV, the score is computed
separately per variant, so variant 1, 2, and 3 may choose different estimators.
Variant 4 chooses `best_global_model` by construction.

### CATE/ITE-focused selection

This selection is used for `global_method_best_cate` and
`variant_method_best_cate`.

The same R2 filter is applied, but the optimization metric is validation
`ite_mse` instead of validation `ate_bias`.

This distinction matters because a model can estimate the population ATE well
while ranking or estimating individual treatment effects poorly. Notebook 06
therefore reports both ATE-selected and CATE-selected results, and the paper
summary table prefers the CATE-selected results when evaluating CATE/ITE MSE.

## Evaluation in Notebook 06

Notebook 06 loads all saved alpha result files and builds combined DataFrames
using `prepare_dataframes_for_analysis()`.

The primary evaluation objects are:

| DataFrame | Meaning |
|---|---|
| `DF_ALL_GLOBAL` | Global method, all estimators, all seeds |
| `DF_ALL_VARIANT` | CDV method, all estimators, all seeds |
| `DF_BEST_GLOBAL` | Global method after validation ATE selection |
| `DF_BEST_VARIANT` | CDV method after validation ATE selection |
| `DF_BEST_GLOBAL_CATE` | Global method after validation ITE MSE selection |
| `DF_BEST_VARIANT_CATE` | CDV method after validation ITE MSE selection |

### ATE metrics

For each method and seed:

```text
true_ate = mean(ite_real)
estimated_ate = mean(ite_pred)
ate_error = estimated_ate - true_ate
ate_squared_error = ate_error^2
```

Across seeds, notebook 06 decomposes ATE MSE as:

```text
bias = mean(ate_error)
variance = var(ate_error)
mse = mean(ate_squared_error)
mse = bias^2 + variance
```

The ATE comparison is run for the best selected estimators and also per
estimator. Statistical testing uses a one-sided paired comparison with the
alternative that CDV has lower MSE than global:

```text
H0: MSE_CDV >= MSE_global
H1: MSE_CDV < MSE_global
```

The tables also report Cohen's d and 95% confidence intervals.

### CATE/ITE metrics

For each individual prediction:

```text
ite_sq_error = (ite_pred - ite_real)^2
ITE MSE = mean(ite_sq_error)
PEHE = sqrt(ITE MSE)
```

Notebook 06 also computes a per-instance bias-variance decomposition across
seeds. For each fixed test position, it compares the mean prediction across
seeds to the true ITE and measures seed-to-seed prediction variance. These are
then averaged over instances.

CATE is evaluated in two ways:

- Per estimator, using `DF_ALL_GLOBAL` and `DF_ALL_VARIANT`.
- Best CATE selection, using `DF_BEST_GLOBAL_CATE` and
  `DF_BEST_VARIANT_CATE`.

### Ranking metrics

Notebook 06 evaluates whether predicted ITEs rank patients similarly to true
ITEs. This is useful for prescriptive targeting even when absolute ITE values
are biased.

It reports:

- Kendall's tau.
- Spearman's rho.

Two ranking scopes are computed:

- **Pooled ranking:** rank all test rows in a seed together.
- **Within-variant weighted ranking:** compute rank quality inside each variant
  and average by variant size.

At `alpha = 0`, all true ITE values are constant, so ranking is undefined.
Notebook 06 keeps these rows visible and marks the ranking metrics as `NaN`.

## Main Results from the Current Analysis

The current notebook 06 paper summary reports:

| alpha | ATE MSE CDV | ATE MSE Global | ATE improvement | CATE MSE CDV | CATE MSE Global | CATE improvement | Kendall tau CDV | Kendall tau Global | Seeds |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.0012 | 0.0021 | 44.9% | 0.0267 | 0.0511 | 47.8% | NA | NA | 13 |
| 0.25 | 0.0022 | 0.0029 | 24.0% | 0.2160 | 0.7568 | 71.5% | 0.7244 | 0.5604 | 13 |
| 0.50 | 0.0022 | 0.0028 | 20.3% | 0.3804 | 0.9705 | 60.8% | 0.7278 | 0.7344 | 12 |
| 0.75 | 0.0014 | 0.0030 | 53.2% | 0.6192 | 1.2207 | 49.3% | 0.8027 | 0.7599 | 12 |
| 1.00 | 0.0018 | 0.0033 | 45.1% | 0.8886 | 1.5100 | 41.2% | 0.8295 | 0.7464 | 12 |

The reported CATE rows use validation ITE MSE selection. Ranking is pooled
Kendall's tau.

The high-level interpretation is:

- CDV reduces ATE MSE across the analyzed alpha values.
- CDV reduces CATE/ITE MSE substantially under the CATE-selected evaluation.
- CDV usually improves treatment-effect ranking, especially at stronger
  heterogeneity levels.
- `alpha = 0` is a special case for ranking because every individual has the
  same true ITE, so rank correlation is mathematically undefined.

## What a New Reader Should Keep in Mind

The experiment is not testing whether CDV has extra information. The global
model sees the same sentinel-coded feature table, so it can observe the process
path through missingness. The test is whether explicitly splitting by observed
decision variants makes the causal response surfaces easier to estimate.

The "others" bucket is not a fully local CDV model. It is a conservative
fallback to the global model because it combines rare patterns with limited
sample size. The meaningful CDV comparison is therefore concentrated in the top
three variants that cover roughly 87% of the population.

The validation split is central to the experiment. It is used to select the
estimator family for each seed and variant, separately from the fixed test split
used for reporting final performance.

The ATE-selected and CATE-selected "best" models answer different questions.
ATE selection asks which model best estimates the population mean treatment
effect. CATE selection asks which model best estimates individual effects. Both
are valid, but they should not be mixed without stating which selection rule was
used.

Finally, the current result files are a partial run relative to the planned
100-seed configuration. The methodology is defined for 100 seeds, but any
numerical claim should specify the number of completed seeds actually loaded by
notebook 06.
