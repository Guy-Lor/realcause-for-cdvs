# Paper Notes — Revised Synthetic Experiment

## Important Points for the Paper

### 1. Structural Equations of the Synthetic DGP

The DGP (`cdv_utils/synthetic_dgp.py`) is unchanged from the original NB05/06
experiment — the revised experiment (`01_experiment.ipynb`) reuses the same
generator and only changes the *protocol* around it (fresh train **and** test
data per outer seed, CDV discovery on training data only, five methods
compared instead of two). This note documents the structural equations
themselves and the rationale behind each design choice, for use when writing
the synthetic-DGP paragraph of the paper.

**Shared exogenous variables** (identical across all subgroups):

$$X_1 \sim \text{Uniform}(0,5), \qquad X_2 \sim \text{Bernoulli}(0.5)$$

**Per-subgroup equations.** Each of the six observed subgroups $v$ defines its
own (i) feature-generation equations for the variables it observes, (ii) a
propensity function $P(T=1\mid \cdot)$, (iii) a baseline-outcome function
$Y(0)=f_v(\cdot)+\varepsilon$, and (iv) a treatment-effect term
$\Delta_v(\cdot)$ with $\tau_v = 5 + \alpha\Delta_v(\cdot)$. All feature draws
use $|\mathcal{N}(\mu,\sigma)|$ (half-normal) to keep values on a
non-negative, measurement-like scale.

| SG | Share | Features (equations, chained) | Propensity logit | $Y(0)$ | $\Delta_v(\cdot)$ |
|----|------:|---|---|---|---|
| SG0a | 20% (½ of 40%) | $V=\lvert\mathcal N(0.9X_1+0.6X_2,1.0)\rvert$; $Z_1=\lvert\mathcal N(0.7V+0.2X_1+0.5,0.7)\rvert$ | $-0.75+0.35V+0.25Z_1-0.20X_1+0.30X_2$ | $50+3X_1+2V+1.5Z_1$ | $2.8V+1.2Z_1-6.0$ |
| SG0b | 20% (½ of 40%) | $Z_1=\lvert\mathcal N(0.8X_1+1.1X_2+0.5,0.7)\rvert$; $V=\lvert\mathcal N(0.6Z_1+0.4X_2+0.2,0.8)\rvert$ | *same as SG0a* | $49+2.4X_1+1.1V+2.4Z_1+1.0X_2$ | $1.8V+1.4Z_1+0.6\,VZ_1/3-5.0$ |
| SG1 | 30% | $E=\lvert\mathcal N(0.8X_1+0.5,1.0)\rvert$; $V=\lvert\mathcal N(X_1+0.3E,1.0)\rvert$ | $-0.3+0.3E+0.2V-0.2X_1$ | $45+2.5X_1+2E+1.8V$ | $-2.8V+2.6E+0.7\,EV/3+2.0$ |
| SG2 | 17% | $V=\lvert\mathcal N(X_1+0.5X_2,1.0)\rvert$; $E=\lvert\mathcal N(0.5X_1+0.3V,0.8)\rvert$; $Z_2=\lvert\mathcal N(0.4E+0.5,0.6)\rvert$ | $-0.4+0.3V+0.2E-0.15Z_2$ | $48+2.8X_1+1.5V+1.2E+Z_2$ | $3.5\sin(V)-2.4E+3.0Z_2+1.0X_2$ |
| SG3 (other) | 5% | $Z_1=\lvert\mathcal N(0.5X_1+1.0,0.8)\rvert$; $Z_2=\lvert\mathcal N(0.3X_2+1.2,0.7)\rvert$ | $-0.2+0.3Z_1-0.2Z_2+0.1X_1$ | $52+3.2X_1+1.8Z_1+1.5Z_2$ | $1.8\,Z_1Z_2/2-2.0$ |
| SG4 (other) | 4% | $E=\lvert\mathcal N(0.6X_1+0.8,1.0)\rvert$; $Z_1=\lvert\mathcal N(0.4X_1+0.3E,0.8)\rvert$ | $-0.3+0.35E+0.2Z_1-0.15X_1$ | $47+2.6X_1+2.2E+1.4Z_1$ | $1.2E^2/3-2.2Z_1+1.0$ |
| SG5 (other) | 4% | $V=\lvert\mathcal N(X_1+0.3X_2,1.0)\rvert$; $Z_1=\lvert\mathcal N(0.3V+0.5,0.7)\rvert$; $Z_2=\lvert\mathcal N(0.2X_1+0.8,0.6)\rvert$ | $-0.4+0.25V+0.2Z_1-0.15Z_2$ | $49+3.0X_1+1.6V+1.3Z_1+Z_2$ | $2.2V-1.8Z_1+2.0Z_2$ |

Propensity is $P(T=1\mid\cdot)=\text{sigmoid}(\text{logit})$. Outcomes are
completed as $Y(0)=f_v(\cdot)+\varepsilon$, $\varepsilon\sim\mathcal
N(0,2.0)$; $Y(1)=Y(0)+\tau_v$; $T\sim\text{Bernoulli}(P(T=1\mid\cdot))$;
observed $Y = (1-T)Y(0)+T\,Y(1)$. The true ITE (used as ground truth for CATE
evaluation) is $\text{ite}=Y(1)-Y(0)=\tau_v$ and is treated as deterministic
in $X$ — no separate treatment-effect noise term is added, since the ITE
itself is the quantity being scored against.

---

### 2. Rationale Behind the Structural-Equation Design

**Chained (not independent) feature generation.** Within a subgroup, later
features are generated as noisy functions of earlier features (e.g.
$X_1,X_2\to V\to Z_1$ in SG0a) rather than drawn independently. This creates a
genuine causal chain — an "iCGraph" — per decision variant, not merely a
different *set* of observed columns. It mimics realistic pre-decision
processes where one measurement (e.g. an abnormal vital sign) triggers a
follow-up test (e.g. a lab panel), so the correlation structure among
pre-decision variables differs meaningfully across variants, not just their
availability.

**SG0 split into two latent mechanisms (SG0a / SG0b) with identical observed
features and identical propensity.** This is the key adversarial case built
into the DGP. SG0a and SG0b expose exactly the same observed feature set
($X_1,X_2,V,Z_1$), so the CDV-discovery heuristic — which groups cases purely
by *observed* feature-availability pattern — cannot and should not tell them
apart; they are pooled into the same CDV. They differ only in their
feature-generation, baseline-outcome, and treatment-effect equations. Sharing
one propensity function of the observed history ($V,Z_1,X_1,X_2$) across the
two mechanisms is deliberate: it preserves the assumption central to the
paper's theory that, conditional on the observed decision history, treatment
selection does not depend on the latent causal mechanism ($T \perp G \mid
S$). This lets the experiment show that CDV modeling helps even though a CDV
is not always a perfectly homogeneous causal unit — some residual
within-CDV heterogeneity is expected and is not a bug, it is a realistic
boundary case.

**Conflicting sign / functional form of the same variable across variants**
(e.g. $V$ has a *positive* effect on the treatment benefit in SG0a but a
*negative* effect in SG1; $V$ enters through $\sin(V)$ in SG2; interaction
terms $V\cdot Z_1$, $E\cdot V$, $Z_1\cdot Z_2$; a quadratic term $E^2$ in
SG4). This operationalizes the paper's central claim: the same observed
variable can carry different, even opposite, causal meaning depending on
which pre-decision path produced it. A pooled global model, given a single
feature table (with sentinel-coded absence), must learn these
variant-specific interactions indirectly through the missingness pattern,
whereas each CDV-specific model only has to learn one simple, low-dimensional
local response surface. This is the mechanism the ATE/CATE MSE comparison is
designed to expose.

**Linear interpolation $\tau_v = 5 + \alpha\Delta_v(\cdot)$, with each
$\Delta_v$ chosen so its subgroup-conditional mean is roughly self-canceling
around the shared baseline.** This gives a single, continuous knob that moves
the DGP from a fully homogeneous world ($\alpha=0$: every case has the same
constant treatment effect, $\tau=5$, regardless of subgroup) to a fully
heterogeneous world ($\alpha=1$: each variant has its own nonlinear response
surface), while holding population shares, feature-generation, propensity,
and baseline-outcome equations fixed across the sweep. This isolates
"strength of causal heterogeneity" as the only variable manipulated across
$\alpha\in\{0,0.25,0.5,0.75,1\}$, so any change in the CDV-vs-global gap as
$\alpha$ increases can be attributed to heterogeneity rather than to a
confounded change in sample composition or noise.

**Independent, subgroup-specific propensity coefficients (rather than one
shared propensity function of all six covariates).** Each decision
path/pathway is allowed its own decision rule, which is realistic (different
clinical pathways may weigh the same measurements differently when deciding
on treatment) and avoids coupling the propensity model's functional form to
the treatment-effect model's functional form, which would make it harder to
attribute estimation error specifically to effect heterogeneity. The
resulting propensities are moderate in magnitude, which combined with the
`OVERLAP_LO`/`OVERLAP_HI`/`OVERLAP_MIN_FRACTION` retention checks in the CDV
discovery step, ensures adequate treatment/control overlap within each
retained CDV.

**Sentinel encoding (`-100`) for structurally absent features, rather than
`NaN` or imputation.** Standard scikit-learn / EconML estimators used for the
global baselines cannot consume `NaN` directly, and mean/model-based
imputation would erase the information that a variable was never measured
for that decision path — which is itself the signal a global model needs (and
that CDV-Separate uses structurally, by dropping the column entirely instead
of encoding its absence). The sentinel value keeps this information explicit
and available to every method on equal footing.

**Skewed population shares (0.40 / 0.30 / 0.17 for the three main variants;
0.05 / 0.04 / 0.04 for the "other" bucket, ≈87% covered by the top three).**
This mirrors the empirical pattern observed in the real Sepsis CDV discovery,
where a small number of dominant missingness patterns cover the large
majority of cases (see `cdv_experiments/sepsis/PAPER_NOTES.md`, point 3).
It also deliberately stress-tests CDV-Separate: even with a large total
`N_TRAIN` (7,000), the minority "other"-bucket-adjacent variants receive only
a few hundred training cases each- it lets the synthetic
experiment illustrate both the benefit of CDV separation (when local sample
size is adequate) and its limits (when a variant is too small).

**Outcome noise is added only to $Y(0)$, not to $\tau$.** The true ITE must
be known exactly for every case in order to compute CATE MSE against ground
truth. Adding idiosyncratic noise to the treatment effect itself would turn
the "true" ITE into a stochastic, unobservable target, which would conflate
estimator error with irreducible noise and defeat the purpose of using a
fully synthetic DGP with known ground truth.

---

### 3. Synthetic ATE MSE Results

The main text reports CATE MSE (Section 6) and points here for the
companion ATE MSE results, computed on the same 20 seeds ×
$\alpha\in\{0,0.25,0.5,0.75,1\}$ grid (`artifacts/primary_summary.csv`).
Values are mean $\pm$ std across seeds.

| $\alpha$ | CDV-Separate (proposed) | Global (Sentinel) | Global + Missingness | Global + Miss. + CDV ID | Matched Random Partitions |
|---:|---|---|---|---|---|
| 0.00 | 0.0074 ± 0.0078 | 0.0049 ± 0.0086 | 0.0051 ± 0.0102 | 0.0053 ± 0.0098 | 0.0040 ± 0.0052 |
| 0.25 | 0.0061 ± 0.0073 | 0.0038 ± 0.0052 | 0.0041 ± 0.0068 | 0.0044 ± 0.0072 | 0.0043 ± 0.0051 |
| 0.50 | 0.0064 ± 0.0087 | 0.0043 ± 0.0040 | 0.0040 ± 0.0046 | 0.0042 ± 0.0056 | 0.0060 ± 0.0059 |
| 0.75 | 0.0079 ± 0.0106 | 0.0053 ± 0.0053 | 0.0046 ± 0.0049 | 0.0044 ± 0.0050 | 0.0084 ± 0.0083 |
| 1.00 | 0.0117 ± 0.0145 | 0.0069 ± 0.0069 | 0.0049 ± 0.0058 | 0.0051 ± 0.0054 | 0.0121 ± 0.0121 |

Paired differences vs. CDV-Separate (from
`artifacts/paper_summary_ate_cate_by_alpha.csv`, significance stars are
raw/unadjusted; an FDR-BH-adjusted version is in
`artifacts/paper_summary_ate_cate_by_alpha_fdr_bh.csv`):

| $\alpha$ | Global (Sentinel) | Global + Missingness | Global + Miss. + CDV ID | Matched Random Partitions |
|---:|---|---|---|---|
| 0.00 | 0.002 ± 0.008 ns | 0.002 ± 0.009 ns | 0.002 ± 0.009 ns | 0.003 ± 0.007 ** |
| 0.25 | 0.002 ± 0.006 * | 0.002 ± 0.006 * | 0.002 ± 0.005 ns | 0.002 ± 0.008 ns |
| 0.50 | 0.002 ± 0.009 ns | 0.002 ± 0.007 * | 0.002 ± 0.006 * | 0.000 ± 0.012 ns |
| 0.75 | 0.003 ± 0.012 ns | 0.003 ± 0.010 * | 0.004 ± 0.009 ** | -0.001 ± 0.016 ns |
| 1.00 | 0.005 ± 0.017 ns | 0.007 ± 0.015 ** | 0.007 ± 0.013 *** | -0.000 ± 0.021 ns |

**Reading.** ATE MSE stays small (≲0.012) and broadly comparable across all
five methods and all $\alpha$, in contrast to the large, monotonically
widening CATE MSE gap reported in the main text. This is expected: every
method still aggregates over the same overall treatment/control mix, and the
subgroup-specific $\Delta_v$ terms are constructed to be roughly
self-canceling in expectation (see point 2 above), so pooling across
subgroups largely washes out the heterogeneity that hurts CATE estimation.
CDV-Separate is not the best-performing method on ATE MSE at any $\alpha$
(its per-subgroup models trade a small amount of aggregate bias/variance for
subgroup-level accuracy) — the ATE task is simply not where CDV separation is
expected to help; its value is specific to CATE/individual-level estimation,
which is why the main text leads with CATE MSE.

---

