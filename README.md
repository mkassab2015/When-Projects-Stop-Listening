# When Projects Stop Listening: Requirements Disengagement as a Longitudinal Precursor to Open-Source Software Abandonment

> **HICSS 2027 Submission** 

This repository contains the complete, reproducible analysis pipeline for the paper *"When Projects Stop Listening: Requirements Disengagement as a Longitudinal Precursor to Open-Source Software Abandonment"*. The study introduces **Requirements Decay** as a four-dimensional construct and provides the first longitudinal matched-cohort evidence of requirements-related deterioration preceding OSS project abandonment.

---

## Key Findings

| Finding | Value |
|---|---|
| Repositories analysed | 89 abandoned + 89 active (178 total) |
| Issues extracted | 31,393 across 24-month observation windows |
| Requirements classified | 12,267 (39.1%) |
| Zero-requirement months (abandoned) | **82.3%** vs 39.5% (active) |
| Roadmap milestone linkage gap | **16× lower** in abandoned (0.7% vs 11.2%) |
| Primary significant finding | `ignored_requirement_ratio`: β̂₃ = 0.012, p_FDR = 0.008 |
| Early-window effect size | Cohen's d_early = 0.505 (months 13–24 before T₀) |
| Sensitivity check (log stars) | β̂₃ = 0.011, p_FDR = 0.009 — holds after size control |

---

## Repository Structure

```
.
├── 00_config.py                        # Constants, domain taxonomy, codebook
├── 01_github_helpers.py                # GitHub API wrappers, rate-limit handling
├── 02_discover_repositories.py         # Candidate discovery and abandonment verification
├── 03_match_repositories.py            # Nearest-neighbour matched cohort construction
├── 04_build_repo_panel.py              # Panel construction, pseudo-T₀ assignment
├── 05_extract_issues_comments.py       # Issue and comment extraction (chunked, auto-resume)
├── 06_auto_classify_requirements.py    # OR-union requirement classifier + type hierarchy
├── 07_create_manual_coding_sample.py   # Stratified validation sample (918 issues)
├── 08_evaluate_and_merge_manual_labels.py  # Classifier validation + LLM label merge
├── 09_compute_negotiation_metrics.py   # Response time, discussion depth, ignored ratio
├── 10_build_repository_month_metrics.py    # Aggregate to repository-month panel
├── 11_extract_commit_controls.py       # Monthly commit counts (control variable)
├── 12_generate_plots.py                # Trend plots with 95% bootstrap CIs (seed=42)
├── 13_mixed_effects_models.py          # Mixed-effects models + BH-FDR correction
├── 14_robustness_checks.py             # Four pre-specified robustness specifications
├── 15_export_manuscript_tables.py      # Export T1–T8 as manuscript-ready CSVs
└── README_RUN_ORDER.md                 # Quick-start run guide
```

All outputs are saved to Google Drive at:
```
/content/drive/MyDrive/requirements_decay_study/
├── raw/            GitHub API extractions (issues, comments, commits)
├── processed/      Classified and derived datasets
├── manual_coding/  LLM validation sample and completed labels
├── results/        Model outputs, tables T1–T8, validation reports
├── figures/        Trend plots and combined four-panel figure
└── checkpoints/    Progress logs for auto-resume
```

---

## Prerequisites

### 1. Google Colab + Google Drive
All scripts are designed to run as Google Colab cells. Outputs are saved to Google Drive after each repository, enabling auto-resume across sessions.

**Drive space required:** ~3 GB for the full 89+89 study.

### 2. GitHub Personal Access Token
Without a token: 60 API requests/hour. Authenticated: 5,000/hour.

Get one at: **GitHub → Settings → Developer settings → Personal access tokens (classic)**
Scope required: `public_repo` only.

### 3. Anthropic API Key (for LLM validation in script 08)
Required only for LLM-assisted classifier validation. Approximate cost: $3–5 for 918 issues using Claude Haiku.

Get one at: **console.anthropic.com**

---

## Quickstart

### Session setup (run at the start of every Colab session)

```python
from google.colab import drive
drive.mount('/content/drive')

import os
os.environ['GITHUB_TOKEN']       = 'ghp_your_token_here'
os.environ['ANTHROPIC_API_KEY']  = 'sk-ant-your-key-here'  # only needed for script 08

%run '/content/drive/MyDrive/requirements_decay_scripts/00_config.py'
%run '/content/drive/MyDrive/requirements_decay_scripts/01_github_helpers.py'
```

Scripts 00 and 01 must be rerun at the start of every new Colab session. They load constants and API helper functions into memory but do not write any data.

### Full pipeline run order

```python
# Stage 1 — Repository discovery and matching
%run 02_discover_repositories.py       # ~3–6 hours for 89+89; auto-resumes
%run 03_match_repositories.py          # ~10 minutes
%run 04_build_repo_panel.py            # ~30–60 minutes

# Stage 2 — Data extraction
%run 05_extract_issues_comments.py     # ~8–16 hours; auto-resumes on disconnect
%run 06_auto_classify_requirements.py  # ~20–40 minutes

# Stage 3 — Validation sample
%run 07_create_manual_coding_sample.py # < 2 minutes
```

**STOP HERE — classifier validation required (see below)**

```python
# Stage 4 — Metrics and analysis
%run 08_evaluate_and_merge_manual_labels.py  # < 2 minutes after validation
%run 09_compute_negotiation_metrics.py        # ~10 minutes
%run 10_build_repository_month_metrics.py     # < 5 minutes
%run 11_extract_commit_controls.py            # ~4–8 hours; auto-resumes
%run 12_generate_plots.py                     # ~5 minutes
%run 13_mixed_effects_models.py               # ~15 minutes
%run 14_robustness_checks.py                  # ~20 minutes
%run 15_export_manuscript_tables.py           # < 2 minutes
```

---

## Classifier Validation (Between Scripts 07 and 08)

Script 07 creates a stratified sample of 918 issues saved to:
```
MyDrive/requirements_decay_study/manual_coding/manual_coding_sample.csv
```

This study used **LLM-assisted validation** (Claude Haiku) with an aligned prompt that instructs the model to apply the same mechanical label and text-signal rules as the automated classifier. The validation yielded:

| Metric | Value |
|---|---|
| Precision | 0.609 |
| Recall | 0.731 |
| F1 | 0.664 |
| Accuracy | 0.686 |
| Per-stratum F1 range | 0.620–0.720 |

The LLM coding script produces `manual_coding_completed.csv`. Save it to:
```
MyDrive/requirements_decay_study/manual_coding/manual_coding_completed.csv
```

**Intra-rater reliability (optional):** Recode a 15% subset after a 2-week gap. Save as `manual_recode_sample.csv` with columns `issue_id` and `manual_is_requirement_recode`. Script 08 computes Cohen's kappa automatically if this file exists.

**Skipping validation:** Script 08 falls back to automated labels throughout if no completed file is found. Acknowledge in Threats to Validity.

---

## Study Design

### Abandonment operationalisation (two-stage verification)
**Stage 1:** No `pushed_at` events for ≥12 months before May 10, 2026 (the fixed study reference date).

**Stage 2 (commit API verification):**
- ≥20 human commits in the 24-month observation window (confirms prior genuine activity)
- Zero commits in the 12 months following T₀ (confirms genuine abandonment)

Of 2,848 metadata-filtered candidates, only 809 (28.4%) passed commit-level verification, confirming that `pushed_at` is an unreliable abandonment proxy.

### Matched cohort construction
1-to-1 nearest-neighbour matching within (language × domain) strata on:
- log(1 + stars)
- log(1 + forks)
- Age in days

**Open issue count is intentionally excluded** as a matching variable — for abandoned repositories it reflects accumulated unanswered issues at data collection time, making it a post-treatment quantity.

**Active cohort verification:** After matching, active repositories are verified to have ≥20 commits within their actual pseudo-T₀ window. 11 pairs failed this check and were dropped, yielding the final sample of **89 abandoned + 89 active** repositories.

**Covariate balance:**
| Variable | SMD | Balance |
|---|---|---|
| log(stars) | −1.309 | Poor (addressed via log_stars covariate in all models) |
| log(forks) | −0.332 | Poor |
| age_days | −0.043 | Good |

### Four decay dimensions and metrics

| Dimension | Primary Metric | Definition |
|---|---|---|
| **Activity** | `requirement_issues` | Count of requirement-like issues per month |
| **Elaboration** | `avg_actionability_score` | 0–4 composite (text ≥30 words, rationale, acceptance, label) |
| **Negotiation** | `avg_discussion_depth` | Mean speaker-role alternations per thread (0 = ignored) |
| **Responsiveness** | `ignored_requirement_ratio` | Proportion of requirements receiving no non-author response |

Additional metrics: `vagueness_density`, `roadmap_issue_ratio`, `avg_first_response_hours` (capped at 8,760h), `requirement_ratio` (secondary — see note below).

> **Note on `requirement_ratio`:** Treated as secondary because it conflates proportion change with total volume change when overall issue activity declines near T₀. The count metric `requirement_issues` is the primary Activity measure.

### Statistical analysis
For each of nine outcomes:

```
Y_rt = β₀ + β₁·A_r + β₂·τ_t + β₃·(A_r × τ_t) + X_rt·γ + u_r + ε_rt
```

Where τ_t is a **reversed** time index (τ=1 at month 24, furthest from T₀; τ=24 at month 1), so a positive β̂₃ indicates accelerating deterioration as T₀ approaches.

**Selective controls:**
- Issue-count outcomes: `log(1 + commits)` only (excluding `log_total_issues` to avoid post-treatment bias)
- Quality/responsiveness outcomes: `log(1 + commits)` + `log(1 + total_issues)`

**Multiple testing:** Benjamini-Hochberg FDR correction across all 9 interaction-term p-values. Primary criterion: p_FDR < 0.05.

**Effect sizes:** Cohen's d computed from repository-level means (clustered) to avoid inflating the effective sample size.

**Robustness specifications (four pre-specified):**
1. Full controls (baseline)
2. No commit control
3. Excluding final 6 months before T₀
4. Well-matched pairs only (distance ≤ 2.0)

---

## Results Summary

### Model results (Table 3 in manuscript)

| Outcome | β̂₃ | p_raw | p_FDR | Sig. | Model |
|---|---|---|---|---|---|
| Req. issues | −0.037 | .300 | .548 | No | Random slopes |
| **Ignored ratio** | **+0.012** | **.001** | **.008** | **Yes** | **Random slopes** |
| Vagueness density | +0.000 | .834 | .834 | No | Random slopes |
| Roadmap ratio | −0.002 | .313 | .548 | No | Random slopes |
| Discussion depth | −0.025 | .142 | .498 | No | Random intercepts |
| Bug expectation | +0.005 | .638 | .744 | No | Random slopes |

Two outcomes (`avg_actionability_score`, `avg_first_response_hours`) could not be modelled due to sparse data. `requirement_ratio` excluded due to degenerate parameter estimates.

### Sensitivity check for star-count imbalance
Re-estimating the ignored_requirement_ratio model with log(1 + stars) as an additional covariate:
- β̂₃ = 0.011, p_FDR = 0.009
- Finding holds — effect is not attributable to the size differential between cohorts.

### Early warning pattern
The effect is stronger in the **early window** (months 13–24 before T₀):
- d_early = 0.505 (medium effect)
- d_late = 0.207 (small effect)

This suggests responsiveness decay is detectable more than a year before visible technical abandonment (commit cessation).

---

## Configuration

Key parameters in `00_config.py`:

```python
N_ABANDONED  = 100     # Target matched pairs per cohort
N_ACTIVE     = 100
OBS_MONTHS   = 24      # Observation window length
RUN_DATE     = '2026-05-10'  # Fixed reference date for reproducibility
MIN_COMMITS_BEFORE_T0 = 20   # Minimum commits required in observation window
MANUAL_VALIDATION_SIZE = 1000  # Issues to sample for classifier validation
```

**Pilot mode:** Set `N_ABANDONED = N_ACTIVE = 10` to verify the pipeline end-to-end before running the full study.

**Scaling up:** After a successful pilot, set `N_ABANDONED = N_ACTIVE = 100`, delete `raw/` and `processed/` folders from Drive (not `results/` or `checkpoints/`), and rerun from script 02.

---

## Reproducibility Notes

- **Study reference date fixed:** May 10, 2026 — all cutoff dates derived from this fixed anchor
- **Bootstrap seed:** `np.random.seed(42)` in script 12 for reproducible confidence intervals
- **Auto-resume:** Scripts 05 and 11 save progress after every repository; rerunning resumes from the next incomplete repository
- **Checkpoint system:** All long-running scripts write checkpoint JSON/CSV files to Drive after each completed repository

---

## Threats to Validity

| Threat | Type | Mitigation |
|---|---|---|
| Star-count imbalance (SMD = 1.309) | Internal validity | log(stars) added as covariate; sensitivity confirmed p_FDR = 0.009 |
| Classifier noise (F1 = 0.664) | Construct validity | Per-stratum F1 range 0.620–0.720; symmetric across cohorts |
| Non-author response as maintainer proxy | Construct validity | Acknowledged; maintainer identity not exposed by GitHub issues API |
| Robustness sparsity (3/4 specs singular) | Conclusion validity | Full-controls spec confirmed; sparse-data limitation documented |
| GitHub-only sample | External validity | Largest public OSS host; diverse language and domain coverage |
| Observational design | Causal inference | Results are associational; no causal claims made |

---

## Citation

If you use this code or data, please cite:

```bibtex
@inproceedings{kassab2027projects,
  title     = {When Projects Stop Listening: Requirements Disengagement as a
               Longitudinal Precursor to Open-Source Software Abandonment},
  booktitle = {Proceedings of the 60th Hawaii International Conference on
               System Sciences (HICSS)},
  year      = {2027},
  note      = {To appear}
}
```

---

## Dependencies

All scripts run in Google Colab with standard libraries. Install the following before running:

```python
!pip install anthropic --break-system-packages    # Script 08 only (LLM validation)
!pip install scikit-learn statsmodels scipy pandas numpy matplotlib seaborn
```

GitHub API access requires Python `requests` (pre-installed in Colab).

---

## License

This repository is released for academic reproducibility. The data extracted via the GitHub API is subject to GitHub's Terms of Service. The manuscript text is copyright of the authors.
