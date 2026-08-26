# EACL Submission Audit

Date: 2026-08-02

Audited artifact: `eacl27_paper_staging/main.pdf`

## A. Executive Summary

当前稿件在格式、匿名性、核心数字和已检查的数据流方面已接近可提交状态。未发现 desk-rejection 级问题、未发现数值冲突，也未在 source-speech-group-held-out 或 interpreter-disjoint 实现中发现数据泄漏。三个主要剩余风险是：训练入口依赖未被 Git 跟踪的 `train_v1.ipynb`；缺少既有 INTERSPEECH 稿件，无法完成文本和贡献重叠的确定性审计；历史训练环境未保存精确 PyTorch/Transformers/scikit-learn patch 版本、GPU 型号和训练时间。匿名稿本身未暴露作者身份。与 INTERSPEECH 的重叠风险暂评为中等，原因是证据不足，而不是已发现重复发表。

### Fact Inventory

| Item | Verified fact |
|---|---|
| Title | Predicting Professional Promptness Ratings in Simultaneous Interpreting from Text and Segment-Onset Delay |
| Submission type | Long paper; numbered main content ends on page 8 |
| PDF | 14 A4 pages: 8 main-content pages, then Limitations/Ethics/Availability, references, and ACL two-column appendices |
| Candidate records | 679 shared evaluator identifiers; 632 complete rating-plus-timing candidates |
| Main cohort | 622 segments after the prespecified 0--20 s onset-range filter |
| Exclusions | 40 missing onset differences, 7 missing R05 promptness, then 8 negative and 2 above-20-s onset differences |
| Groups and people | 16 source-speech groups, 7 interpreters, 2 professional evaluators |
| Evaluator recruitment and compensation | Recruited through a professional interpreting company; RMB 1,000 per hour |
| Directions | Zh-to-En: 505; En-to-Zh: 117 |
| Target | Equal-weight mean of two independent integer 0--3 professional promptness ratings; aggregate range .5--3.0 |
| Platform constraint | LQ=0 forces EXP=0 and promptness=0; 17/1,244 evaluator-segment triples, affecting 17/622 aggregate segments |
| Main protocol | 16 deterministic outer source-speech-group folds plus four inner source-group partitions for cross-fitted quality features |
| Generalization audit | Seven-fold leave-one-interpreter-out; source speeches may overlap through other interpreters |
| Main systems | delay; predicted LQ+EXP; predicted LQ+EXP+delay; lexical/structural+delay; full model; same-rubric human-quality reference |
| Sensitivities | linear/nonlinear delay, alpha, clipping, bounded regression, training-fold mean, residualization, rater, direction, and subgroup audits |
| Metrics | Pearson r, Spearman rS, MSE; MAE and calibration in uncertainty/range audits |
| Random seeds | 20260718, 20260719, 20260720 |
| Supplement | Appendices A-E are included after references; no separate identifying link is present |
| Public artifacts | No public/anonymous repository link in the review PDF; camera-ready release is planned subject to governance |
| INTERSPEECH evidence | No prior INTERSPEECH manuscript was available in the audited checkout |

## B. Prioritized Issue Table

| Priority | Severity | Category | File/Location | Problem | Evidence | Recommended fix | Auto-fixed |
|---|---|---|---|---|---|---|---|
| P1 | Major | Reproducibility | `run_train_v1.py`, `.gitignore` | A clean clone cannot execute the formal training launcher because it dynamically loads `train_v1.ipynb`, which exists locally but is ignored and untracked. | `git ls-files train_v1.ipynb` is empty; `.gitignore` matches `*.ipynb`; launcher opens the notebook at runtime. | Move the formal training definitions into a tracked Python module, or explicitly track a sanitized notebook and test from a clean clone. | No |
| P1 | Major | Prior-work overlap | Project-wide | The requested INTERSPEECH overlap audit cannot be completed without the prior manuscript and its exact dataset/contribution statement. | No INTERSPEECH PDF/source is present; the current paper contains no self-identifying prior-work wording. | Provide the prior manuscript for text/claim comparison; cite and disclose shared data/models in third person if applicable. | No |
| P2 | Moderate | Reproducibility | Appendix E; historical logs | Exact PyTorch, Transformers, and scikit-learn patch versions, GPU model, fold training time, checkpoint revision, and truncation counts are not preserved. | Requirements give ranges; COMET is fixed at 2.2.4; remote training logs identify Python 3.10 behavior but no complete environment snapshot. | Freeze a tested lockfile/container now and document that it is a reproduction environment; record hardware for any rerun. | Partly: absence is now disclosed |
| P2 | Moderate | Supplement packaging | Appendix B/E; staging directory | The paper promises fold manifests, counts, provenance, checksums, and environment files, but the anonymous staging directory does not yet contain a complete packaged release. | Appendix describes planned camera-ready contents; no consolidated exact fold-manifest file is tracked in the paper directory. | Build an anonymous supplement/archive containing only permitted aggregate artifacts and exact manifests. | No |
| P2 | Moderate | Baseline completeness | Methods/Results | Direct text-to-promptness artifacts exist but are not reported in the current paper, leaving a predictable reviewer question about the need for staged quality prediction. | Three direct-text and text+delay seed summaries exist under `experiments/aaai_direct_lat_summary_seed_*`. | Author decision: add a compact appendix comparator or explain why it is excluded from the claimed comparison set. | No |
| P2 | Moderate | Ethics/data rights | Ethics; release plan | IRB, interpreter consent, evaluator recruitment and compensation, rights restrictions, and re-identification risk are reported, but the eventual release license is not documented in the audited evidence. | Section 3.1 records evaluator compensation at RMB 1,000 per hour; no release-license record was found. | Select a license compatible with consent and institutional governance. | Partly: compensation is now documented |
| P3 | Minor | Construct provenance | Limitations | It remains unknown whether pilot calibration examples overlapped the analyzed cohort. | The limitation is explicitly stated; no identifiable calibration-candidate list is available. | Confirm from project records if possible; otherwise retain the current limitation. | No |
| P3 | Minor | Anonymous packaging | Git metadata | The working repository remote contains the author's GitHub username, although the manuscript and PDF do not. | `git remote -v` shows an identifying URL; PDF/source scan is clean. | Do not include `.git`, local README/history, or identifying URLs in the anonymous submission archive. | No |
| P3 | Minor | Compilation environment | Temporary local TeX toolchain | Tectonic emits an invalid UTF-8 warning from an old temporary `lineno.sty`; this is not manuscript source. | Build succeeds; no undefined citations/references, errors, or overfull boxes. | Compile once with the official current author-kit environment before upload. | No |

## C. Numerical Consistency Table

Canonical source: `experiments/eacl_paper_canonical_20260728/paper_results.json`.

| Quantity | Current value | Locations checked | Conflict status |
|---|---:|---|---|
| Main cohort | 622 | Abstract, Data, tables, appendix, canonical JSON | Consistent |
| Source groups / interpreters / evaluators | 16 / 7 / 2 | Abstract, Data, Methods, appendix | Consistent |
| Direction counts | 505 / 117 | Data, direction results, generated table | Consistent |
| Delay-only Pearson | .318 | Abstract, Table 3, Figure 2, Results, JSON | Consistent |
| Predicted LQ+EXP | .435 +/- .037 | Table 3, JSON/macros | Consistent; `+/-` is seed SD |
| Predicted quality+delay | .475 +/- .026 | Abstract, Table 3, Figure 2, Results | Consistent; `+/-` is seed SD |
| Structural+delay | .632 | Abstract, Table 3, Figure 2, Results | Consistent |
| Full model | .658 +/- .010 | Abstract, Table 3, Figure 2, Results, Conclusion | Consistent |
| Same-rubric human-quality reference | .809 | Table 3, Figure 2, Results | Consistent; not described as a ceiling |
| Primary exact-swap statistic | Delta r=.1697, p=.000687 | Methods, Results, canonical JSON | Consistent |
| Seed-level point delta | .1566 [.0761,.2434] | Results/macros, canonical JSON | Consistent |
| Secondary full-minus-structure delta | .030 [.002,.063]; Delta MSE=-.012 [-.026,.000] | Results, appendix, JSON | Consistent and framed descriptively |
| Cross-rater | .430 and .543 | Results, Appendix Table 7, correction log, JSON | Consistent |
| Superseded cross-rater | .502 and .828 | Appendix provenance note and correction log only | Intentional historical note |
| Promptness agreement | Pearson .259; ICC(2,1)=.236 | Data, Results, appendix, JSON | Consistent |
| LOIO delay / quality+delay | .048 / .377 +/- .080 | Results, Appendix Table 8, Figure 2 | Consistent |
| LOIO structure / full | .511 / .573 +/- .019 | Results, Appendix Table 8, Figure 2 | Consistent |
| LOIO full centered / macro | .340 +/- .012 / .358 +/- .005 | Results, appendix, JSON | Consistent |

`Delta r=.1697` is not expected to equal `.475-.318=.157`: the table averages seed-level correlations, whereas the exact swap test first averages each segment's predictions across seeds and then computes one nonlinear Pearson difference.

No stale `.8066` or `.8369` result was found. The `.337` value in Appendix Table 9 is the valid Zh-to-En direction-specific delay correlation, not an obsolete pooled result.

## D. Leakage Audit

**Verdict: No leakage found in the checked protocols and artifacts.**

- Every outer source-speech group is excluded from quality supervision, development selection, and second-stage promptness fitting before test prediction.
- Inner source-group OOF predictions provide quality features for every second-stage training row; no row receives quality predictions from a model trained on that row's LQ/EXP label.
- Scaling is fit on the outer promptness-training partition only; fixed hyperparameters and delay knots do not use outer-test labels.
- Exact duplicate source-text/interpreted-output pairs are removed from upstream train/dev relative to each outer test group.
- In LOIO, all output from the held-out interpreter is excluded from quality training, development, checkpoint selection, and promptness fitting.
- Source overlap in LOIO is explicitly acknowledged, so the paper does not claim jointly unseen speech and interpreter transfer.
- Rater-specific models fit the stated target rater only on outer-training groups.

Residual limitation: a clean-room rerun from tracked files is currently blocked by the ignored notebook, so this verdict is an implementation/artifact audit rather than an independently reconstructed training run.

### Data-flow summary

```text
outer entity held out
  -> remove it from quality train/dev and checkpoint selection
  -> split remaining speech groups into four inner partitions
  -> train inner quality models and generate OOF LQ/EXP for promptness training
  -> train final outer-excluded quality model and predict the outer entity
  -> fit scaler + Ridge only on inner-OOF promptness-training rows
  -> access outer labels once for final evaluation
```

## E. INTERSPEECH Overlap Audit

**Risk rating: Medium / unable to fully verify.**

The current manuscript presents a distinct task: predicting professional promptness ratings and decomposing timing, structural, and automatically estimated quality signals under nested held-out protocols. Its title, abstract, contribution list, main tables, rater analysis, structural baseline, and LOIO audit are centered on promptness rather than generic interpreting-quality estimation. No large copied block or self-identifying phrase is visible within the current staging files.

However, the prior INTERSPEECH manuscript, its source, and an explicit table of shared records/models were not available. Therefore text reuse, exact cohort overlap, and contribution reuse cannot be ruled out. Before submission, compare title/abstract/introduction/contributions, dataset description, model architecture, tables, and conclusions side by side. If data or the upstream quality estimator are shared, cite the prior work normally in third person and state exactly what is reused and what is new.

## F. Applied Changes

- `sections/3_task_data.tex`: documented the platform rule `LQ=0 => EXP=0 and promptness=0` and clarified that lower-bound dimensions are not independent.
- `sections/3_task_data.tex`: documented professional-company recruitment and evaluator compensation at RMB 1,000 per hour.
- `sections/7_limitations.tex`: added the low-end association caveat caused by the platform constraint.
- `sections/A_appendix.tex`: added the 17/1,244 platform-rule audit, a complete compact quality-estimator model card, the exact loss, and readable fixed-position case/model-card tables.
- `sections/4_method.tex`: replaced the nonexistent anonymous-model-card reference with Appendix D.
- `sections/E_reproducibility.tex`: added Python/library requirements, compute provenance, training parameters/seeds, bootstrap/permutation implementation, and planned release contents.
- `main.tex`: included Appendix E after references while preserving ACL two-column layout.
- `audit_eacl_paper_consistency.py`: added deterministic checks for the platform rule, model card, and reproducibility section.
- `experiments/eacl_paper_canonical_20260728/paper_results.json` and generated tables/macros: regenerated from one canonical results source.
- Appendix layout: fixed the C/D/E ordering and eliminated overfull boxes without changing main-content pagination.

## G. Remaining Manual Actions

1. Make the formal training pipeline runnable from a clean clone by tracking or replacing `train_v1.ipynb`.
2. Provide the prior INTERSPEECH paper for a definitive overlap audit and add a transparent third-person relationship statement if data/models overlap.
3. Select and verify the intended release license; evaluator compensation is now documented.
4. Freeze a tested exact environment and record hardware for any reproducibility rerun.
5. Build the anonymous supplement/archive with exact outer/inner manifests, per-fold counts, checksums, correction log, provenance map, and permitted aggregate predictions.
6. Decide whether to report the existing direct text-to-promptness comparator in an appendix.
7. Confirm whether pilot examples overlapped the analysis cohort if records allow.
8. Compile once with the official current EACL/ARR author kit before upload.

## H. Final Submission Checklist

### Must Fix Before Claiming Full Reproducibility

- [ ] Formal training runs from a clean tracked checkout.
- [ ] Exact anonymous fold/provenance package is assembled and tested.

### Strongly Recommended

- [ ] Prior INTERSPEECH overlap review completed and relationship disclosed where applicable.
- [ ] Exact tested software lock/container and hardware note added.
- [ ] Release-license terms confirmed against consent and governance constraints.
- [ ] Direct text-to-promptness comparator inclusion decided.
- [ ] Pilot-example overlap checked if possible.

### Confirm in Submission System

- [ ] Select long-paper track and verify current EACL/ARR page policy.
- [ ] Upload the anonymous PDF only; exclude `.git`, identifying URLs, raw media, identity mappings, credentials, and restricted row-level outputs.
- [ ] Confirm IRB/consent wording and data-availability answers match institutional records.
- [ ] Confirm supplementary files are anonymous and permitted by the data-use agreement.

### Passed

- [x] ACL review template is used.
- [x] A4 page size.
- [x] Numbered main content ends on page 8.
- [x] Limitations section is present.
- [x] References precede appendices.
- [x] Appendix remains ACL two-column format.
- [x] All fonts are embedded.
- [x] PDF opens, copies, and searches normally.
- [x] No undefined citations or references.
- [x] No LaTeX errors or overfull boxes.
- [x] No TODO/FIXME/highlight placeholders in manuscript source.
- [x] Manuscript/PDF anonymity scan passed.
- [x] Canonical numerical consistency audit passed.
- [x] No leakage found in checked protocols.

## Final Build Record

- Build: successful with Tectonic and a local TeX search bundle.
- PDF: A4, 14 pages, 328,359 bytes, PDF 1.5.
- SHA-256: `6edb8e386596df88f4cd53e5f3e1b78c45dbc08319dd71dad2c8054d7af7fce3`.
- Main-content boundary: Conclusion ends on page 8; Limitations begins on page 9.
- Warnings: underfull boxes and one invalid UTF-8 warning from the temporary old `lineno.sty`; no manuscript-source blocking warning.
