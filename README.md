# Requirements Decay Study — Run Guide (v3, all fixes applied)

All outputs saved to Google Drive:
```
/content/drive/MyDrive/requirements_decay_study/
├── raw/           Raw API extractions
├── processed/     Cleaned and derived datasets
├── manual_coding/ Manual classification files
├── results/       Model outputs, tables, validation reports
├── figures/       Trend plots and combined panels
└── checkpoints/   Progress logs (JSON + CSV)
```

---

## Initial Colab setup

```python
from google.colab import drive
drive.mount('/content/drive')

import os
os.environ['GITHUB_TOKEN'] = 'ghp_your_token_here'
```

A classic GitHub token with `public_repo` scope is sufficient.
Without a token: 60 API requests/hour. Authenticated: 5,000/hour.

---

## Run order

```python
%run 00_config.py
%run 01_github_helpers.py
%run 02_discover_repositories.py      # ~30–90 min for 100+100
%run 03_match_repositories.py
%run 04_build_repo_panel.py
%run 05_extract_issues_comments.py    # longest step; auto-resumes
%run 06_auto_classify_requirements.py
%run 07_create_manual_coding_sample.py
```

---

## STOP FOR MANUAL CLASSIFICATION

Open:
```
MyDrive/requirements_decay_study/manual_coding/manual_coding_sample.csv
```
Fill the `manual_*` columns using the codebook printed by script 07.
Save as:
```
MyDrive/requirements_decay_study/manual_coding/manual_coding_completed.csv
```

### Intra-rater reliability (strongly recommended)
Recode 15% of issues after a 2-week gap. Save as:
```
MyDrive/requirements_decay_study/manual_coding/manual_recode_sample.csv
```
Required columns: `issue_id`, `manual_is_requirement_recode`
Script 08 computes Cohen's kappa automatically.

### Skipping manual classification
Script 08 falls back gracefully to auto labels only and logs a
limitations note. Acknowledge explicitly in manuscript Threats to Validity.

---

## Continue

```python
%run 08_evaluate_and_merge_manual_labels.py
%run 09_compute_negotiation_metrics.py
%run 10_build_repository_month_metrics.py
%run 11_extract_commit_controls.py    # auto-resumes on disconnect
%run 12_generate_plots.py
%run 13_mixed_effects_models.py
%run 14_robustness_checks.py
%run 15_export_manuscript_tables.py
```

---

## Pilot vs full study

In `00_config.py`:
```python
N_ABANDONED = 10   # → change to 100 for full study
N_ACTIVE    = 10   # → change to 100 for full study
```
Run the full pipeline at N=10 first. After it succeeds end-to-end,
delete `raw/` and `processed/` and rerun from script 02 with N=100.

---

## All fixes applied (v3)

| Fix | Severity | Script | Description |
|---|---|---|---|
| Word-boundary domain matching | Moderate | 00 | `\b` regex prevents 'ml'→'email', 'ai'→'trail' etc. |
| **Broken regex patterns fixed** | **Serious** | **00** | **All malformed `\b(^\|[^a-z])X([^a-z]\|$)\b` replaced with clean `\bX\b`** |
| Codebook embedded in config | Moderate | 00, 07 | No external file path dependency in Colab |
| Active cohort verification moved to correct window | **Serious** | **02, 04** | **Verification now uses pseudo-T0 window (not own pushed_at). Pairs where active repo fails are dropped and logged.** |
| Endogenous control removed | Serious | 13, 14 | `log_total_issues` excluded from issue-count outcome models |
| BH-FDR multiple testing correction | Critical | 13, 14 | Applied across all 9 outcomes; `p_interaction_fdr` is primary criterion |
| `discussion_depth` floor fix | Serious | 09 | Zero for ignored issues; was incorrectly returning 1 |
| Bootstrap CI seeded | Minor | 12 | `np.random.seed(42)` for reproducible figures |
| `first_response_hours` theoretically capped | Minor | 09 | Capped at 8,760h (1 year); theoretically motivated; per-cohort safe |
| `include_groups=False` in groupby.apply | Minor | 09 | Prevents pandas ≥2.2 deprecation warning/error |
| Cohen's d uses repo-level means | Moderate | 13 | Respects clustered structure; prevents inflated effect sizes |
| T4 t-test disclaimer | Moderate | 15 | Repeated-measures t-tests labelled DESCRIPTIVE_ONLY |
| T5 pairwise correlations | Moderate | 15 | Pairwise complete obs replaces listwise dropna; avoids selection bias |
| T6 reports raw + FDR p-values | Critical | 15 | Both columns reported; FDR is primary |
| `requirement_ratio` caveat in T6 | Moderate | 15 | Printed note flagging ratio instability; count metric is primary |
| T7 FDR per robustness spec | Critical | 14, 15 | BH-FDR applied within each specification |
| Robustness checks now test all 9 outcomes | Moderate | 14 | Aligned with script 13; FDR pools comparable |
| `no_commit_control` spec corrected | Moderate | 14 | Issue-count models now use `log_total_issues` in this spec (not empty) |

---

## Key methodological decisions for manuscript

| Decision | Where documented |
|---|---|
| Abandonment verified by commit count (zero post-T0, ≥MIN pre-T0) | Script 02 |
| Active cohort verified by commit count (≥MIN in window) | Script 02 |
| Domain inferred from GitHub topics via word-boundary regex | Script 00 |
| Matching: (language × domain) strata, NN on log_stars/log_forks/age | Script 03 |
| Balance table (SMD) proves cohort comparability | Script 03 |
| 1000-result cap handled by recursive date-window splitting | Script 01 |
| OR-classifier: high recall, precision validated in script 08 | Script 06 |
| `bug_expectation` separate type — not merged with feature | Script 06 |
| `discussion_depth = 0` for ignored issues (floor corrected) | Script 09 |
| `first_response_hours` winsorised at 99th pct | Script 09 |
| `log_total_issues` excluded for issue-count outcomes (post-treatment) | Script 13 |
| BH-FDR correction (9 outcomes) — primary significance criterion | Script 13 |
| Random slopes compared to intercepts by AIC | Script 13 |
| Cohen's d on repo-level means (clustered) | Script 13 |
| 4 robustness checks with FDR within each specification | Script 14 |
| T4 t-tests flagged as descriptive only | Script 15 |
| Calendar-time confound (pseudo-T0) acknowledged and mitigated | Script 04 |

---

## Threats to validity (notes for manuscript)

1. **Construct validity of requirements identification**: OR-classifier favours
   recall over precision. Mitigated by manual validation (script 08).

2. **Active cohort temporal confound**: active repos receive their abandoned
   partner's T0 as pseudo-T0, which may not align with their actual activity
   peak. Mitigated by monthly_commits control and age/stars matching.

3. **Maintainer identity proxy**: "non-author response" used because maintainer
   status is not reliably exposed by the GitHub issues API without extra calls.

4. **GitHub-centric sample**: results may not generalise to GitLab, Bitbucket,
   or proprietary platforms.

5. **Domain inference reliability**: topics are user-supplied and inconsistent
   across repositories. Mitigated by word-boundary matching and 'other' fallback.

6. **Multiple comparisons**: BH-FDR correction applied but does not eliminate
   the possibility of false positives, particularly for outcomes that survive
   correction with p-values close to the q=0.05 threshold.
