# Current methods and reproducibility record

This document describes the regenerated repository as of 11 November 2026. The original Word documents are retained unchanged as historical records. Current run-specific numbers come from the generated outputs.

## 1. Historical preparation

`scripts/preprocess.py` aligns the six raw inputs to calendar quarters. Monthly unemployment and prices require three observed months. Year-over-year inflation is calculated before discarding 1999. The inflation benchmark is uniformly 2%, including before 2012; this is a comparison convention. Historical unemployment gaps use time-varying NROU. Missing early SPF forecasts do not remove otherwise usable Okun observations.

The panel contains 104 quarters. The baseline regression uses 103 complete unemployment/output pairs, 2000Q1–2025Q3. The complete four-gap sample contains 75 observations. Sources and transformations are detailed in [the data dictionary](../data/README.md).

## 2. Okun calibration

`scripts/calibrate_okun.py` replaces the missing `okun_calibration_save.do` referenced in the original declaration. The supplied repository did not include that Stata file; this Python implementation reproduces the stated estimates, without claiming to reconstruct the original program.

```text
output_gap = 0.1563712903 - 0.9469595754 * unemployment_gap + residual
n = 103
OLS slope SE = 0.0458682
Newey–West slope SE = 0.0601190
```

The HAC covariance uses a Bartlett kernel with four quarterly lags and the n/(n−k) finite-sample correction, with k=2 including the intercept. Excluding 2020 yields a slope of −0.953943; restricting to complete observations on all four gaps yields −0.935856. These are descriptive robustness checks, not causal identification.

The screen uses the 2.5th and 97.5th empirical residual percentiles: **[−1.5946056336, 1.6573401543]**, inclusive. NumPy's `averaged_inverted_cdf` reproduces the supplied thresholds. For this n=103 sample neither target falls on an exact empirical jump, so `inverted_cdf` gives the same values. The default linear interpolation gives [−1.578467, 1.628151] and is not the selected convention. The six-decimal constants in the old generator agree with the reproduced calibration after rounding. The current generator reads full-precision calibration JSON, avoiding a second manually maintained parameter set.

## 3. Scenario design

The Cartesian grid is:

| Gap | Levels |
| --- | --- |
| Inflation, pp | −1, 0, 1, 2 |
| Unemployment, pp | −1.5, −0.5, 0.5, 1.5 |
| Expectations, pp | −0.5, 0, 0.5, 1 |
| Output, percent | −3, −1.5, 0, 1.5, 3 |

There are 320 initial candidates. The Okun screen retains 144; sampling six per inflation–unemployment cell with seed **42** gives 96. Sort by all four gaps before assigning S001–S096; add all-zero S000. The 97-row scenario CSV is byte-identical to the pre-organization file, verified by its archived checksum.

The coverage plot and descriptive statistics include S000; group coverage excludes it. Fourteen cells cover every eligible expectations level, and fifteen cover every eligible output level. Marginal balance for those two variables is not imposed. The unemployment–output correlation is approximately −0.740. The screen constrains that relationship only; it does not establish that every four-variable combination is jointly realistic.

`presentation_order.csv` saves the seed-**2026** shuffle of all 97 scenarios. This is a reusable ordering artifact, not an executed prompt schedule. Future held-out experiments must respect the Task 2 split.

## 4. Policymaker ground truth

```text
latent_rate_bps = alpha + beta_pi * inflation_gap
                 - beta_u * unemployment_gap
                 + beta_exp * expectations_gap + beta_y * output_gap
```

| Type | alpha | beta_pi | beta_u | beta_exp | beta_y | beta_pi / beta_u |
| --- | --- | --- | --- | --- | --- | --- |
| A | 0 | 25 | 5 | 8 | 2 | 5 |
| B | 0 | 20 | 10 | 8 | 2 | 2 |
| C | 0 | 15 | 15 | 8 | 2 | 1 |
| D | 0 | 10 | 20 | 8 | 2 | 0.5 |
| E | 0 | 5 | 25 | 8 | 2 | 0.2 |

All types share beta_pi + beta_u = 30 and common expectations/output coefficients. This isolates composition of preferences. Rounded actions use a **5 bp** step, ties away from zero, clipped to [−100, 100]. The current sample does not activate clipping. A fine grid preserves sensitivity to the weaker slopes, but is a modeling choice requiring review; it is not a claim that this is the usual policy increment.

The surface figure shows continuous rules at zero expectations and output gaps. It is a slice of the rule and includes grid coordinates outside the screened sample.

## 5. Split, neutral control, and histories

S000 is held separately from the 96-row base. Its unique inflation–unemployment cell cannot support within-cell splitting. Under the current choice, it appears in `neutral_control.csv` with a zero action for all five types; it enters neither histories nor the 48-state test set. This is a provisional design decision for review.

Within each of 16 cells, a seeded shuffle assigns three scenarios to training and three to testing. The algorithm searches seeds 0–999, minimizing the sum of standardized condition numbers and five times expectations/output marginal imbalance, with large penalties for rank or required-region failures. Selection uses scenario features, not policy outcomes. The current seed is **775**, score **5.4035**. Selection is deterministic for the recorded input and environment; a changed input may select a different seed.

All types share a 20-scenario training history. The first two contrast tightening and easing. Later selection prioritizes rank, scaled distance, new levels, and region coverage, with a condition-number penalty. Each cell contributes at most two history states. Prefixes 2/5/10/20 form nested histories. The first five cover tightening, easing, and boundary regions and have rank five including the intercept. Two observations alone cannot identify all five coefficients.

Region definitions are evaluated in order: tightening if inflation >= 1 and unemployment <= −0.5; easing if inflation <= 0 and unemployment >= 0.5; otherwise boundary if abs(inflation − unemployment) <= 1; remaining states are other. Tags do not alter decisions and must stay outside prompts.

## 6. Current verification results

| Type | True ratio | Recovered from all 96 rounded outcomes | Recovered from 48 test outcomes |
| --- | --- | --- | --- |
| A | 5 | 4.942 | 4.744 |
| B | 2 | 1.993 | 1.984 |
| C | 1 | 0.993 | 0.990 |
| D | 0.5 | 0.498 | 0.494 |
| E | 0.2 | 0.197 | 0.198 |

These are OLS recovery diagnostics on synthetic ground truth. Continuous outcomes recover the specified coefficients, and rounded outcomes preserve strict A–E ratio ordering. Test recovery for Type A is imperfect, so ordering preservation does not imply exact coefficient recovery. No LLM was evaluated.

The generator checks rank, overlap, coverage, coefficient recovery, action bins, and ordering. `verify_results.py` independently checks saved-table consistency, scenario inputs, history membership, shared sequences, nested prefixes, rounding, and all held-out targets. Successful pipeline runs write an environment/hash manifest. Numeric tables should reproduce under the pinned environment; plot bytes can depend on platform/fonts.

## 7. Changes from the supplied project

- Standardized paths and string IDs; fixed Task 2 input location and nonnumeric-ID handling.
- Preserved raw inputs and both original Word documents. Archived the duplicate `preprocess_new.py` as text so it cannot be mistaken for a runnable entry point.
- Added a Python calibration step, saved presentation order and diagnostics, regenerated Task 2 outputs, and added end-to-end verification.
- Regenerated the historical panel from raw files; floating-point serialization can differ from the supplied processed CSV. The calibration matches the declaration. Initial hashes are retained in `docs/archive/original_checksums.json`; keys identify files at the start of reorganization.
- The old Task 2 note used seed 21 and different run-specific numbers. This run uses the current unchanged Task 1 sample and seed 775. Its first five regions are tightening, easing, boundary, boundary, boundary. The original note's reported five-region sequence and recovered ratios are historical results, not the current run.
- Original declaration paths and Python-version statements are historical; README commands and the manifest describe the current implementation.

## 8. Decisions before locking materials

Review the 5 bp action step, relative coefficient weights, the scenario screen, and S000's separate-control role. After approval, regenerate from the agreed scenario file, record the approved input hash, and deliberately update provisional labeling in code and documentation together. Then implement prompt-field whitelists, experimental conditions, and the pilot.
