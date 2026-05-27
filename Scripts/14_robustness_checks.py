# 14_robustness_checks.py
# Purpose:
#   Run four robustness checks on the main mixed-effects model results.
#
# Fixes in this version:
#
#   FIX 1 — Endogenous control fix carried through (SERIOUS):
#     Robustness checks now use the same selective controls as script 13:
#     issue-count outcomes use log_commits only; quality/responsiveness
#     outcomes use log_commits + log_total_issues.
#
#   FIX 2 — BH-FDR correction applied within each robustness specification:
#     Within each specification (across all outcomes), raw p-values are
#     FDR-corrected. The robustness summary reports whether findings survive
#     FDR correction in ALL four specifications.
#     Note: robustness checks are descriptive/confirmatory, not the primary
#     inferential tests, so FDR is applied separately per specification rather
#     than pooled across all specifications × outcomes.
#
#   Retained from previous version:
#   - Four specifications: full controls, no commit control, exclude last
#     6 months, well-matched pairs only.
#   - Robustness pivot table.

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

monthly = pd.read_csv(PROCESSED / 'repository_month_metrics_with_controls.csv')

monthly['is_abandoned']     = (monthly['cohort'] == 'abandoned').astype(int)
monthly['time_to_t0_rev']   = OBS_MONTHS - monthly['months_before_t0'] + 1
monthly['log_commits']      = np.log1p(monthly['monthly_commits'])
monthly['log_total_issues'] = np.log1p(monthly['total_issues'])
monthly['poor_match_flag']  = monthly['poor_match_flag'].fillna(False).astype(bool)

# Selective controls (same logic as script 13)
ISSUE_COUNT_OUTCOMES = {'requirement_issues', 'requirement_ratio', 'bug_expectation_issues'}

# FIX: all 9 outcomes from script 13 included here for consistency.
# Previously 3 outcomes were omitted without justification, making the
# FDR pools in T6 and T7 incomparable. All 9 are now tested in all specs.
OUTCOMES = [
    'requirement_issues',
    'requirement_ratio',
    'ignored_requirement_ratio',
    'avg_actionability_score',
    'avg_vagueness_density',
    'roadmap_issue_ratio',
    'avg_first_response_hours',
    'avg_discussion_depth',
    'bug_expectation_issues',
]

# (label, row_filter, issue_count_controls, quality_controls)
SPECIFICATIONS = [
    ('full_controls',
     None,
     '+ log_commits',
     '+ log_commits + log_total_issues'),

    # FIX: 'no_commit_control' previously left issue-count outcome models
    # with NO controls at all (empty string), making it uninterpretable as a
    # sensitivity test on the commit control specifically.
    # For issue-count outcomes we now use log_total_issues as the alternative
    # control (acknowledging the endogeneity caveat in this sensitivity context);
    # for quality/responsiveness outcomes we drop log_commits only.
    # This makes the spec a clean test of whether the commit control drives results.
    ('no_commit_control',
     None,
     '+ log_total_issues',            # FIX: was '' (no controls); now consistent alternative
     '+ log_total_issues'),

    ('exclude_last_6_months',
     monthly['months_before_t0'] > 6,
     '+ log_commits',
     '+ log_commits + log_total_issues'),

    ('well_matched_pairs_only',
     ~monthly['poor_match_flag'],
     '+ log_commits',
     '+ log_commits + log_total_issues'),
]

all_rows = []

for spec_label, row_filter, ic_controls, qr_controls in SPECIFICATIONS:
    spec_rows = []

    for metric in OUTCOMES:
        if metric not in monthly.columns:
            continue

        dfm = monthly.copy()
        if row_filter is not None:
            dfm = dfm[row_filter]

        controls = ic_controls if metric in ISSUE_COUNT_OUTCOMES else qr_controls
        dfm = dfm[['repo','is_abandoned','time_to_t0_rev',
                   'log_commits','log_total_issues', metric]
                 ].replace([np.inf,-np.inf], np.nan).dropna()

        n = len(dfm)
        if n < 50:
            spec_rows.append({'outcome': metric, 'specification': spec_label,
                              'n': n, 'status': 'skipped'})
            continue

        formula = f'{metric} ~ is_abandoned * time_to_t0_rev {controls}'
        try:
            res  = smf.mixedlm(formula, data=dfm, groups=dfm['repo']).fit(
                reml=False, method='lbfgs')
            coef = res.params.get('is_abandoned:time_to_t0_rev', np.nan)
            p    = res.pvalues.get('is_abandoned:time_to_t0_rev', np.nan)
            spec_rows.append({
                'outcome': metric, 'specification': spec_label,
                'n': n, 'n_repos': dfm['repo'].nunique(),
                'controls': controls,
                'coef_interaction': round(coef, 4),
                'p_raw': round(p, 4),
                'p_fdr': np.nan,   # filled below
                'sig_raw': p < 0.05,
                'sig_fdr': False,  # filled below
                'aic': round(res.aic, 2),
                'status': 'ok',
            })
        except Exception as e:
            spec_rows.append({'outcome': metric, 'specification': spec_label,
                              'n': n, 'status': f'error:{e}'})

    # FIX 2: apply BH-FDR within this specification
    ok = [r for r in spec_rows if r.get('status') == 'ok']
    if ok:
        raw_ps = [r['p_raw'] for r in ok]
        _, p_adj, _, _ = multipletests(raw_ps, alpha=0.05, method='fdr_bh')
        for r, pa in zip(ok, p_adj):
            r['p_fdr']  = round(pa, 4)
            r['sig_fdr'] = pa < 0.05

    all_rows.extend(spec_rows)

robustness = pd.DataFrame(all_rows)
save_csv_checkpoint(robustness, RESULTS / 'robustness_checks.csv', '14_robustness')

# ── Summary pivot: robust if FDR-significant in all four specs ─────────────────
print('\nROBUSTNESS SUMMARY (FDR-corrected within each specification)')
print('="*70')

ok_df = robustness[robustness['status'] == 'ok'].copy()
pivot = ok_df.pivot_table(index='outcome', columns='specification',
                           values='sig_fdr', aggfunc='first')
pivot['robust_all_specs_fdr'] = pivot.all(axis=1)

# Also show raw significance for reference
pivot_raw = ok_df.pivot_table(index='outcome', columns='specification',
                               values='sig_raw', aggfunc='first')
pivot_raw.columns = [f'{c}_raw' for c in pivot_raw.columns]

summary = pivot.join(pivot_raw)
display(summary)

display(robustness[['outcome','specification','n','controls',
                     'coef_interaction','p_raw','p_fdr','sig_fdr','status']])
