# 15_export_manuscript_tables.py
# Purpose:
#   Export all manuscript-ready tables.
#
# Fixes in this version:
#
#   FIX 1 — T4 t-test disclaimer added (MODERATE):
#     Table T4 descriptive statistics previously reported Welch t-tests
#     between cohorts using raw monthly observations, which are (a) not
#     independent (repeated measures per repo) and (b) uncorrected for
#     multiple comparisons. The t-test p-values are retained for orientation
#     but now carry a clear disclaimer that they are descriptive only and
#     must NOT be used as the primary inferential evidence. The mixed-effects
#     model results (T6) are the primary tests.
#
#   FIX 2 — T6 now reports both raw and FDR-corrected p-values, and flags
#     sig_fdr_q05 as the primary significance column.
#
#   FIX 3 — T7 reports FDR-corrected significance per robustness specification.

import numpy as np
import pandas as pd
from scipy import stats

monthly = pd.read_csv(PROCESSED / 'repository_month_metrics_with_controls.csv')
issues  = pd.read_csv(PROCESSED / 'issues_with_negotiation_metrics.csv')
monthly['is_abandoned'] = (monthly['cohort'] == 'abandoned').astype(int)

# ── T1: Dataset summary ────────────────────────────────────────────────────────
t1 = monthly.groupby('cohort').agg(
    repositories=         ('repo',                     'nunique'),
    repo_months=          ('repo',                     'count'),
    total_issues=         ('total_issues',             'sum'),
    requirement_issues=   ('requirement_issues',       'sum'),
    avg_monthly_commits=  ('monthly_commits',          'mean'),
    avg_requirement_ratio=('requirement_ratio',        'mean'),
    avg_ignored_ratio=    ('ignored_requirement_ratio','mean'),
    pct_zero_req_months=  ('requirement_issues', lambda x: (x==0).mean()),
).reset_index().round(3)

save_csv_checkpoint(t1, RESULTS / 'T1_dataset_summary.csv')
print('T1 — Dataset summary:'); display(t1)

# ── T2: Covariate balance ──────────────────────────────────────────────────────
bal = RESULTS / 'covariate_balance_table.csv'
if bal.exists():
    print('\nT2 — Covariate balance:'); display(pd.read_csv(bal))
else:
    print('[WARNING] T2 missing — run script 03.')

# ── T3: Requirement type distribution ─────────────────────────────────────────
type_col = 'final_req_type' if 'final_req_type' in issues.columns else 'auto_req_type'
req = issues[issues.get('final_is_requirement',
                         issues.get('auto_is_requirement', 0)) == 1].copy()
t3 = req.groupby(['cohort', type_col]).size().reset_index(name='n')
t3['pct'] = t3.groupby('cohort')['n'].transform(
    lambda x: (x / x.sum() * 100).round(1))
save_csv_checkpoint(t3, RESULTS / 'T3_requirement_types.csv')
print('\nT3 — Requirement type distribution:'); display(t3)

# ── T4: Descriptive statistics with disclaimer ─────────────────────────────────
# FIX: t-test p-values included for orientation only.
# These tests use repeated monthly observations (not independent) and are
# uncorrected for multiple comparisons. They should NOT be interpreted as
# primary inferential evidence. Use T6 (mixed-effects models) for inference.
METRIC_COLS = [
    'requirement_issues','requirement_ratio','ignored_requirement_ratio',
    'avg_actionability_score','avg_vagueness_density','roadmap_issue_ratio',
    'avg_first_response_hours','avg_discussion_depth','monthly_commits',
]
desc_rows = []
for metric in METRIC_COLS:
    if metric not in monthly.columns: continue
    ab_v = monthly[monthly['cohort']=='abandoned'][metric].dropna()
    ac_v = monthly[monthly['cohort']=='active'][metric].dropna()
    _, p = stats.ttest_ind(ab_v, ac_v, equal_var=False)
    for cohort, vals in [('abandoned', ab_v), ('active', ac_v)]:
        desc_rows.append({
            'metric': metric, 'cohort': cohort, 'n': len(vals),
            'mean': round(vals.mean(), 3), 'sd': round(vals.std(), 3),
            'median': round(vals.median(), 3),
            'min': round(vals.min(), 3), 'max': round(vals.max(), 3),
            'ttest_p_DESCRIPTIVE_ONLY': round(p, 4),
        })
t4 = pd.DataFrame(desc_rows)
save_csv_checkpoint(t4, RESULTS / 'T4_descriptive_statistics.csv')
print('\nT4 — Descriptive statistics:')
print('NOTE: ttest_p_DESCRIPTIVE_ONLY — repeated measures, uncorrected.')
print('      Primary inference: see T6 (mixed-effects models).')
display(t4)

# ── T5: Inter-construct correlation matrix ─────────────────────────────────────
# FIX: Use pairwise complete observations (min_periods=10) instead of listwise
# deletion. Listwise dropna() on all four columns simultaneously would exclude
# all zero-issue months (where most metrics are NaN), producing a highly
# selected subsample of only high-activity months and biasing correlations
# upward. Pairwise correlations use all available data for each pair of metrics.
DECAY_DIMS = ['requirement_issues','avg_actionability_score',
              'avg_discussion_depth','ignored_requirement_ratio']
for cohort in ['abandoned','active']:
    sub  = monthly[monthly['cohort']==cohort][DECAY_DIMS]
    # pairwise=True: each cell uses all rows where both variables are non-missing
    corr = sub.corr(method='pearson', min_periods=10).round(3)
    # Record how many pairs contribute to each cell
    n_pairs = sub.notna().T.dot(sub.notna())
    corr.to_csv(RESULTS / f'T5_intercorrelation_{cohort}.csv')
    n_pairs.to_csv(RESULTS / f'T5_intercorrelation_{cohort}_n.csv')
    print(f'\nT5 — Inter-construct correlations ({cohort}) [pairwise]:'); display(corr)
    print(f'  N per pair:'); display(n_pairs)

# ── T6: Main model results (FDR-corrected) ─────────────────────────────────────
mp = RESULTS / 'mixed_effects_results.csv'
if mp.exists():
    t6 = pd.read_csv(mp)
    # FIX: display both raw and FDR-corrected; flag sig_fdr_q05 as primary
    cols = ['outcome','controls','model_type','n_observations','n_repositories',
            'coef_is_abandoned','p_is_abandoned',
            'coef_interaction','p_interaction_raw','p_interaction_fdr',
            'sig_fdr_q05','cohens_d_early','cohens_d_late']
    cols = [c for c in cols if c in t6.columns]
    save_csv_checkpoint(t6[cols], RESULTS / 'T6_model_results.csv')
    print('\nT6 — Main model results (★ = significant at FDR q<.05):')
    display(t6[cols])
    print('\nPrimary significance criterion: p_interaction_fdr (BH-FDR q<.05)')
    print('Uncorrected p_interaction_raw reported for transparency only.')
    print()
    print('NOTE — requirement_ratio interpretation caveat:')
    print('  requirement_ratio = requirement_issues / total_issues.')
    print('  When total_issues declines near T0 (as it does for abandoned repos),')
    print('  this ratio conflates proportion change with volume change.')
    print('  Treat requirement_issues (count) as the cleaner primary Activity metric.')
    print('  requirement_ratio results are reported as secondary/supplementary.')
else:
    print('[WARNING] T6 missing — run script 13.')

# ── T7: Robustness checks (FDR-corrected per specification) ───────────────────
rp = RESULTS / 'robustness_checks.csv'
if rp.exists():
    t7 = pd.read_csv(rp)
    ok = t7[t7['status']=='ok'][['outcome','specification','n','controls',
                                   'coef_interaction','p_raw','p_fdr','sig_fdr']]
    save_csv_checkpoint(ok, RESULTS / 'T7_robustness_checks.csv')
    print('\nT7 — Robustness checks (sig_fdr = BH-FDR within specification):')
    display(ok)
else:
    print('[WARNING] T7 missing — run script 14.')

# ── T8: Classifier validation ──────────────────────────────────────────────────
cp = RESULTS / 'classifier_validation_overall.csv'
if cp.exists():
    t8 = pd.read_csv(cp, index_col=0).round(3)
    save_csv_checkpoint(t8, RESULTS / 'T8_classifier_validation.csv')
    print('\nT8 — Classifier validation:'); display(t8)
else:
    print('[INFO] T8 not available (manual validation not performed).')

print(f'\nAll tables exported to: {RESULTS}')
checkpoint('15_complete', {'results': str(RESULTS)})
