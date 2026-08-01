# Reproducibility and Correction Log

## 2026-07-28: Cross-rater estimand correction

The earlier working values `R05 -> R06 = .5016` and `R06 -> R05 = .8283`
were produced by `run_aaai_reviewer_cpu_analyses.py::cross_rater_cv`. That
routine trained a same-rater mapping on the outer-training groups
(`R05 LQ/EXP + delay -> R05 promptness`, or the R06 analogue), then replaced
both the quality-feature scale and the promptness target with the other rater
only at held-out evaluation. It therefore measured evaluator-scale transport
under a test-time domain swap, not the ordered quality-rater-to-target-rater
association named in the manuscript.

The corrected audit in `run_eacl_rater_calibration_audit.py` fits each ordered
pair directly within every outer-training split. For example,
`R05 LQ/EXP + delay -> R06 promptness` is trained against R06 promptness and
evaluated against R06 promptness on the held-out source-speech group. The
canonical corrected values are:

| Quality inputs | Promptness target | Pearson r |
|---|---:|---:|
| R05 | R05 | .908648 |
| R05 | R06 | .429825 |
| R06 | R05 | .543262 |
| R06 | R06 | .552092 |

The numerical change is due to this estimand and model-definition correction.
It is not caused by adding delay, changing the corrected 622-segment cohort,
changing the 16 source-speech-group folds, changing promptness target values,
or repairing train/test leakage. Both the old and corrected analyses use the
corrected cohort, piecewise delay, and source-speech-group-held-out evaluation.

Canonical files:

- `../experiments/eacl_rater_calibration_audit_20260728/rater_calibration_results.json`
- `../experiments/eacl_rater_calibration_audit_20260728/rater_calibration_predictions.csv`
- `../experiments/eacl_paper_canonical_20260728/paper_results.json`

The old result artifact is retained for provenance but is superseded:

- `../experiments/aaai_reviewer_cpu_corrected_20260721/aaai_reviewer_cpu_results.json`

## 2026-07-28: Residualization interpretation correction

Raw predicted quality and quality residualized against structural features can
span closely related linear feature spaces when the structural features remain
in the downstream Ridge model. Near-identical raw and residualized full-model
predictions are therefore treated as a representation audit, not independent
evidence for a neural quality construct. The revised audit additionally reports
residualized-quality-only prediction, prediction of structure+delay promptness
residuals from quality residuals, and held-out partial R-squared.

Canonical files:

- `../experiments/eacl_structural_neural_audit_20260728_r2/structural_neural_results.json`
- `../experiments/eacl_structural_neural_audit_20260728_r2/structural_neural_predictions.csv`

## 2026-07-29: Interpreter-disjoint aggregation-label correction

The final consistency audit found that two interpreter-disjoint prose values
paired macro-interpreter and within-interpreter-centered means with SDs from a
different aggregation field. The pooled full-model result remains
`r = .573 +/- .019`. The corrected subgroup summaries are full-model centered
`r = .340 +/- .012` and macro-interpreter `r = .358 +/- .005`; predicted-quality
only and predicted-quality-plus-delay macro results are `.229 +/- .019` and
`.221 +/- .024`. These are sample SDs across the same three fixed seeds.

The correction changes only the reported subgroup SD labels. It does not alter
predictions, pooled correlations, per-interpreter values, folds, or model
definitions. Canonical values come from the aggregate fields in:

- `../experiments/aaai_loio_structural_audit_20260722/loio_structural_audit.json`

## 2026-07-30: Complete-case cohort flow and negative-delay range

The normalized professional archive contains 679 segment identifiers with
records from both R05 and R06. Of these, 632 have complete LQ, EXP, and
promptness ratings from both evaluators plus a numeric archived onset
difference. Forty shared identifiers lack onset differences for both
evaluators, and seven lack an R05 promptness rating. The primary cohort retains
622 of the 632 complete candidates under the 0--20 second onset-range rule.

The ten range exclusions comprise eight negative values and two values above
20 seconds. Re-auditing the normalized JSON confirms that the actual negative
range is `-95.295` to `-1.149` seconds; the two upper values are `100` and
`304.331` seconds. The earlier manuscript range `-11.799` to `-1.149` omitted
two more extreme negative records, even though both belong to the 632 complete
candidate set. This correction changes only cohort-flow reporting: all ten
records were already excluded before construction of the unchanged 622-row
analysis cohort.

The `-95.295` and `-93.799` values predate the filename--identifier namespace
repair: both are already present in `profess_eval.json` and
`profess_eval_delay_enriched.json`. The corrected minimum is therefore not a
consequence of the provenance repair; it is a correction of the manuscript's
previously incomplete reported range.

Canonical source:

- `../data/evaluation/profess_eval_delay_enriched_namespace_corrected.json`
