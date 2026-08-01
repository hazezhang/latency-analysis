# Professional Promptness Prediction in Simultaneous Interpreting

This repository contains the paper source, analysis code, evaluation scripts, and
canonical aggregate results for predicting professional promptness ratings from
interpreted text and archived segment-onset delay.

The current manuscript is:

> **Predicting Professional Promptness Ratings in Simultaneous Interpreting from
> Text and Segment-Onset Delay**

## Repository layout

- `eacl27_paper_staging/`: current EACL paper source, generated tables, figures,
  correction log, and compiled PDF.
- `experiments/eacl_paper_canonical_20260728/`: canonical manuscript result source.
- `experiments/eacl_rater_calibration_audit_20260728/`: rater-aware audit outputs.
- `experiments/eacl_structural_neural_audit_20260728_r2/`: structural and neural
  representation audit outputs used by the current paper.
- `build_eacl_paper_results.py`: regenerates the paper's numerical macros and tables.
- `audit_eacl_paper_consistency.py`: checks manuscript values against the canonical
  result definitions.
- `run_*.py`, `prepare_*.py`, and `summarize_*.py`: experiment preparation,
  evaluation, and reporting scripts retained for provenance.

## Environment

Create an isolated Python environment and install the pinned project requirements:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The quality estimator uses the gated `Unbabel/wmt22-cometkiwi-da` checkpoint.
Request access from Hugging Face and provide credentials through the `HF_TOKEN`
environment variable. Tokens are never committed to this repository.

## Reproduce the manuscript tables

```bash
python build_eacl_paper_results.py
python audit_eacl_paper_consistency.py
```

The consistency audit should finish with:

```text
PASS: EACL paper matches canonical result definitions and values
```

## Data and model artifacts

Raw recordings, transcripts, evaluator files, interpreter identity mappings,
downloaded model weights, and training checkpoints are intentionally excluded.
They are subject to consent, licensing, and re-identification constraints and are
not required to inspect the checked-in aggregate results. The paper's availability
and ethics sections describe the planned controlled release.

## Paper build

The LaTeX source is under `eacl27_paper_staging/`. A standard ACL-compatible TeX
installation can build it from that directory. The checked-in `main.pdf` is the
verified 14-page version corresponding to the restored submission draft.
