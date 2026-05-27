# When Projects Stop Listening

Reproduction materials for an empirical study of **human-oriented explanation traces** in public GitHub repositories before and after the mainstreaming of generative AI coding assistants. The study mines commit messages and opened issues from ~800 longitudinally active repositories (Jan 2021 – Apr 2026) and estimates temporal changes around a January 2023 intervention point using interrupted time-series and repository fixed-effects models.

> **Causal scope.** This is an observational, quasi-experimental design. It does not establish that generative AI caused any observed change. The January 2023 intervention point is a *temporal landmark* corresponding to the mainstreaming of ChatGPT and adjacent tools, not an exogenous shock applied to repositories.

---

## Repository layout

```
.
├── Scripts/                                   # 23 executable cells (one per notebook cell)
│   ├── cell_01_install_libraries.py
│   ├── cell_02_imports_utilities.py
│   ├── cell_03_interactive_configuration.py
│   ├── cell_04_authenticate_and_autosave_setup.py
│   ├── cell_05_define_text_metrics.py
│   ├── cell_06_query_candidate_repositories.py
│   ├── cell_07_enrich_metadata_checkpointed.py
│   ├── cell_08_final_repository_sample.py
│   ├── cell_09_upload_repo_list_to_bigquery.py
│   ├── cell_10_extract_commit_messages.py
│   ├── cell_11_extract_issues.py
│   ├── cell_12_compute_commit_metrics.py
│   ├── cell_13_compute_issue_metrics.py
│   ├── cell_14_create_monthly_panel.py
│   ├── cell_15_descriptive_statistics.py
│   ├── cell_16_effect_sizes_tests.py
│   ├── cell_17_interrupted_time_series_models.py
│   ├── cell_18_fixed_effects_sensitivity.py
│   ├── cell_19_generate_trend_figures.py
│   ├── cell_20_create_excel_workbook.py
│   ├── cell_21_create_methodological_audit_log.py
│   ├── cell_22_package_and_download.py
│   └── cell_23_download_key_files.py
├── figures-…/figures/                         # Per-metric ITS trend plots (PNG, 300 dpi)
├── results-…/results/                         # Tables, panel data, audit log, Excel workbook
└── README.md
```

Each `cell_*.py` file corresponds to one cell in the original Colab notebook and is intended to be executed in numerical order. Cells share state via top-level Python variables, so they must run in the same Python session (e.g., as sequential `%run` calls in a single Colab notebook).

---

## Environment

The pipeline is designed to run **entirely in Google Colab**. It requires:

- A Google Cloud project with **BigQuery enabled** and billing attached (for queries against the public `githubarchive.month.*` tables)
- A **GitHub personal access token** with `public_repo` scope (for metadata enrichment of candidate repositories via the REST API)
- Optionally, **Google Drive** mounted for autosave (recommended; protects against Colab disconnects)

Python packages installed by `cell_01_install_libraries.py`:

`google-cloud-bigquery`, `pandas-gbq`, `pyarrow`, `tqdm`, `textstat`, `statsmodels`, `scipy`, `scikit-learn`, `openpyxl`, `xlsxwriter`.

No source code is cloned. The study uses GH Archive event payloads (commit messages, issue titles, issue bodies) and GitHub REST metadata only.

---

## Run order

In a Colab notebook, execute the cells sequentially:

```python
%run cell_01_install_libraries.py
%run cell_02_imports_utilities.py
%run cell_03_interactive_configuration.py      # prompts for project ID, GitHub token, dates
%run cell_04_authenticate_and_autosave_setup.py
%run cell_05_define_text_metrics.py
%run cell_06_query_candidate_repositories.py   # BigQuery, GH Archive
%run cell_07_enrich_metadata_checkpointed.py   # GitHub REST API; auto-resumes from pickle checkpoint
%run cell_08_final_repository_sample.py
%run cell_09_upload_repo_list_to_bigquery.py   # creates temp dataset for join
%run cell_10_extract_commit_messages.py        # BigQuery, PushEvent payloads
%run cell_11_extract_issues.py                 # BigQuery, IssuesEvent payloads (action='opened')
%run cell_12_compute_commit_metrics.py
%run cell_13_compute_issue_metrics.py
%run cell_14_create_monthly_panel.py
%run cell_15_descriptive_statistics.py
%run cell_16_effect_sizes_tests.py             # Mann–Whitney U + Cliff's delta
%run cell_17_interrupted_time_series_models.py # OLS w/ HC3, segmented regression
%run cell_18_fixed_effects_sensitivity.py      # adds C(repo_name) fixed effects
%run cell_19_generate_trend_figures.py         # one PNG per metric
%run cell_20_create_excel_workbook.py          # consolidated workbook
%run cell_21_create_methodological_audit_log.py
%run cell_22_package_and_download.py           # zip + browser download
%run cell_23_download_key_files.py             # individual file downloads
```

Cell `03` is interactive and prompts for configuration values. Defaults used to produce the released results:

| Parameter | Default | Notes |
|---|---|---|
| `study_start_yyyymm` | `202101` | Jan 2021 |
| `study_end_yyyymm` | `202604` | Apr 2026 |
| `intervention_yyyymm` | `202301` | Jan 2023 — temporal landmark |
| `candidate_repo_target` | `3000` | initial pool from BigQuery |
| `final_repo_target` | `800` | analyzed sample |
| `min_pre_months_active` | `6` | activity threshold pre-intervention |
| `min_post_months_active` | `6` | activity threshold post-intervention |
| `min_total_events` | `50` | combined PushEvent + IssuesEvent |
| `random_seed` | `2027` | reproducibility |

The exact configuration is persisted as `study_config.json` alongside the results.

---

## Data sources

1. **GH Archive** (`githubarchive.month.*` via BigQuery): monthly event tables (`PushEvent`, `IssuesEvent`) for the study window. Pull requests are excluded from issue extraction by filtering on `payload.issue.pull_request.url IS NULL`.
2. **GitHub REST API** (`/repos/{owner}/{repo}`): metadata enrichment (stars, language, owner type, fork flag, archived flag, creation/update timestamps). Authenticated requests honour `X-RateLimit-Reset` headers and back off automatically.

---

## Sampling

`cell_06` queries GH Archive for repositories that:

- Were active in at least 6 calendar months before January 2023, and at least 6 months after,
- Had ≥ 50 combined `PushEvent` + `IssuesEvent` events across the full window,
- Had ≥ 3 issue-opening events in both pre- and post-intervention periods,
- Are returned in deterministic order via `FARM_FINGERPRINT(repo_name, seed)`.

The candidate pool (target: 3,000) is then enriched via REST metadata (`cell_07`) and filtered in `cell_08` to drop forks, archived repos, and metadata-failed rows. The final sample is 800 repositories, drawn with `random_state = 2027`.

The realised sample (`final_repository_sample.csv`) contains 800 repositories with 21 metadata columns.

---

## Operationalisation of explanation traces

Defined in `cell_05_define_text_metrics.py`. All markers are **substring matches over lowercased, whitespace-normalised text**.

**Commit message metrics** (per commit):

| Metric | Definition |
|---|---|
| `commit_words` | word count of the full commit message |
| `commit_is_generic` | 1 if the subject line matches one of: `fix(es\|ed)?`, `update(s\|d)?`, `change(s\|d)?`, `misc`, `wip`, `cleanup`, `minor`, `bugfix`, `stuff`, `test` (anchored regex) |
| `commit_has_rationale` | 1 if the message contains any of: *because, due to, so that, in order to, avoid, prevent, support, enable, refactor, improve, reduce, handle, resolve, ensure, simplify, optimize, deprecate* |

**Issue metrics** (per opened issue, title + body concatenated):

| Metric | Definition |
|---|---|
| `issue_total_words` | word count of title + body |
| `issue_has_repro` | contains any of: *steps to reproduce, reproduce, reproduction, minimal example, mre, to reproduce* |
| `issue_has_expected_actual` | contains any of: *expected behavior, actual behavior, expected, actual result, what happened* |
| `issue_has_environment` | contains any of: *environment, version, os, operating system, browser, python, node, java, npm, pip, platform* |
| `issue_has_vague_marker` | contains any of: *not working, doesn't work, does not work, broken, help, error, bug, problem, issue* |

> **No manual validation of these markers was performed.** The dictionaries are intentionally simple, transparent, and reproducible, but they are unvalidated against human judgement. They should be read as proxies, not ground truth — see Threats to Validity.

Item-level metrics are aggregated to **repository × month** means/ratios (`cell_12`, `cell_13`) and merged into a complete repo-month panel covering all months in the study window (`cell_14`), with `post_ai_period` and `time_after_intervention` derived from `yyyymm`.

---

## Analysis

`cell_15` produces descriptive statistics (per-period means, medians, SD, IQR).

`cell_16` computes pre/post comparisons with **Mann–Whitney U tests** and **Cliff's δ** effect sizes on the eight repo-month outcome metrics.

`cell_17` fits **interrupted time-series segmented regressions** of the form:

```
y ~ month_index + post_ai_period + time_after_intervention
```

estimated by OLS with HC3 heteroscedasticity-robust standard errors. Outcomes:

- `mean_commit_words`, `generic_commit_ratio`, `rationale_commit_ratio`
- `mean_issue_words`, `repro_ratio`, `expected_actual_ratio`, `environment_ratio`, `vague_marker_ratio`

`cell_18` re-estimates each model with **repository fixed effects** (`+ C(repo_name)`) as a sensitivity check. This absorbs all time-invariant repo-level heterogeneity.

`cell_19` produces one trend plot per outcome (monthly mean over the study window, with a vertical line at the intervention month).

---

## Outputs

The `results-*/results/` and `figures-*/figures/` directories contain the released artefacts. The key files are:

**Configuration and provenance**
- `study_config.json` — exact parameters used to produce the results
- `methodological_audit_log.txt` — human-readable summary of the pipeline

**Sample**
- `final_repository_sample.csv` — 800 rows, repo-level metadata

**Panel data**
- `repo_month_panel.csv` — 51,200 repo-month rows (800 × 64 months) with all metrics and time variables
- `commit_monthly_repo_metrics.csv` — 22,494 non-empty repo-months for commit metrics
- `issue_monthly_repo_metrics.csv` — 17,317 non-empty repo-months for issue metrics

**Statistical results**
- `prepost_descriptive_statistics.csv` — period means/medians/SD/IQR per metric
- `prepost_effect_sizes_tests.csv` — Mann–Whitney U + Cliff's δ
- `interrupted_time_series_results.csv` — coefficients, SE, p-values for the ITS models
- `interrupted_time_series_model_summaries.txt` — full statsmodels summaries
- `repository_fixed_effects_results.csv` — ITS with `C(repo_name)` absorbing repo-level intercepts
- `artifact_counts.csv` — number of items and contributing repositories per artefact

**Workbook**
- `HICSS2027_results_tables.xlsx` — all of the above as labelled sheets

**Figures** (`trend_*.png`, 300 dpi)
- `trend_mean_commit_words.png`, `trend_generic_commit_ratio.png`, `trend_rationale_commit_ratio.png`
- `trend_mean_issue_words.png`, `trend_repro_ratio.png`, `trend_expected_actual_ratio.png`, `trend_environment_ratio.png`, `trend_vague_marker_ratio.png`

---

## Summary of findings

In the released ITS estimates, the immediate-level shift at the January 2023 boundary (`post_ai_period`) is **not statistically significant for any of the eight outcomes** at conventional thresholds. Several **post-intervention slope** terms (`time_after_intervention`) are positive and significant, indicating that mean issue length, reproduction-step markers, expected/actual markers, and environment markers *rose* over the post-2023 period rather than declined. Commit-message metrics (mean words, generic ratio, rationale ratio) show no significant intervention-period change.

The findings are consistent with an interpretation in which human-oriented explanation traces did **not** collapse in the post-2023 window — and, on several measures, became more frequent. We do not interpret this as a causal claim about generative AI; alternative explanations (template adoption, issue-form rollout on GitHub, changing repository composition, secular trends) are not excluded by the design.

---

## Known issues

- **`post_ai_period_x` / `post_ai_period_y` in `repo_month_panel.csv`.** The monthly panel merges in `cell_14` produce two duplicated copies of the `post_ai_period` column (one from the commit-monthly frame, one from the issue-monthly frame) before the canonical `post_ai_period` column is recomputed from `yyyymm`. The canonical column is the one used by downstream analysis cells; the `_x` and `_y` copies are merge artefacts and should be ignored.
- **Marker overlap.** `issue_has_vague_marker` includes very common substrings (`error`, `bug`, `problem`, `issue`) and will fire on many well-articulated issue bodies. It is best read as a *prevalence* indicator, not a *vagueness* classifier. See Threats to Validity.
- **Heavy tails.** Several outcome distributions (notably `mean_commit_words` and `mean_issue_words`) are highly right-skewed with very large kurtosis. The ITS specifications are estimated in levels with HC3 robust SEs; log-transformed and trimmed-mean robustness checks are not included in the released results.

---

## Threats to validity

1. **Construct validity of the markers.** All eight outcome metrics are computed from dictionary/regex substring matches that were not validated against human judgement. The marker lists are transparent and editable (`cell_05_define_text_metrics.py`), but they conflate surface presence of keywords with the underlying constructs they are meant to proxy (rationale, reproducibility, environment specificity, vagueness). No inter-rater agreement or held-out human-labelled gold set was constructed.
2. **Intervention validity.** January 2023 is a *temporal landmark* selected from the history of public generative-AI tooling, not an exogenous shock applied to the sampled repositories. Any secular trend that turns at or near this point will produce the same pattern in the ITS coefficients. Without a credible control group, observed level/slope changes cannot be attributed to AI adoption.
3. **Sample composition.** Sampling conditions on activity in both pre- and post-windows, which selects for *surviving* repositories and excludes projects that were abandoned during the window. This is a known source of survivorship bias in longitudinal mining studies.
4. **Platform-level confounders.** GitHub introduced and refined issue forms / issue templates over the study period. Increases in structured-marker ratios (e.g., `expected_actual_ratio`, `environment_ratio`) may partly reflect platform UI changes rather than developer behaviour.
5. **GH Archive coverage.** Event payloads are public-stream snapshots and may miss edits, deletions, or private interactions. Commit messages are taken from `PushEvent` payload arrays, which truncate above 20 commits per push; very large pushes are under-sampled.
6. **External validity.** The sample is restricted to public GitHub repositories with sufficient longitudinal activity. Findings do not generalise to private repositories, GitLab/Bitbucket, or proprietary in-house development.
7. **No AI-assistance detection.** The pipeline does not classify repositories or contributions by whether they were AI-assisted. Comparisons are between time periods, not between AI-using and non-AI-using projects.

---

## Reproduction

1. Open a fresh Google Colab notebook.
2. Copy the cells from `Scripts/` into the notebook in numerical order (one cell per notebook cell), or run them via `%run` against a checked-out copy of this repository.
3. Run `cell_03` and provide your own Google Cloud project ID and GitHub token; accept the listed defaults to reproduce the released sample.
4. Run `cell_04` through `cell_23` in order. The full pipeline takes several hours depending on BigQuery quota and GitHub rate limits. Metadata enrichment (`cell_07`) is checkpointed and safely resumable.

The configuration file `study_config.json` records the exact parameters used to produce the released results.

---

## References

The methodology draws on standard practice in empirical software engineering and mining software repositories:

- ACM SIGSOFT Empirical Standards — https://www2.sigsoft.org/EmpiricalStandards/
- Empirical Standards for Repository Mining — https://doi.org/10.1145/3524842.3528032
- Kalliamvakou et al., *The Promises and Perils of Mining GitHub* — https://doi.org/10.1145/2597073.2597074
- GH Archive — https://www.gharchive.org/
- GitHub REST API rate limits — https://docs.github.com/en/rest/using-the-rest-api/rate-limits-for-the-rest-api

---

## License and citation

License and citation information will be added on publication.
