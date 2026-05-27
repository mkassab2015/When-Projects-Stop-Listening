# 13_mixed_effects_models.py
# Purpose:
#   Fit longitudinal mixed-effects models testing whether abandoned repos
#   show different decay trajectories compared to active controls.
#
# Fixes in this version:
#
#   FIX 1 — Multiple testing correction (CRITICAL):
#     Nine outcomes are tested simultaneously. Without correction the familywise
#     error rate is ~37%. Benjamini-Hochberg (BH) FDR correction is applied
#     across all interaction-term p-values. Both uncorrected and FDR-corrected
#     p-values are reported. Findings are flagged as significant only if they
#     survive FDR correction at q < 0.05. This is the primary significance
#     criterion for the manuscript. Uncorrected p-values are reported for
#     transparency and comparison with prior work.
#
#   FIX 2 — Endogenous control variable removed from issue-count models (SERIOUS):
#     log_total_issues is post-treatment for outcomes that are themselves derived
#     from issue counts (requirement_issues, requirement_ratio,
#     bug_expectation_issues). Including it in those models introduces
#     post-treatment bias. The control is now applied selectively:
#       - Issue-count outcomes:  only log_commits as control.
#       - Quality/responsiveness outcomes: log_commits + log_total_issues.
#     This is a principled modelling decision justified in the manuscript.
#
#   FIX 3 — Cohen's d uses repository-level means, not month-level observations (MODERATE):
#     Month-level observations are clustered within repositories (each repo
#     contributes 12 observations per window half). Using raw month-level data
#     underestimates between-group variance and inflates d. We now compute d
#     using one aggregated value per repository per window half, which respects
#     the clustered structure and gives a conservative, defensible effect size.
#
#   Retained from previous version:
#   - Random slopes + random intercepts, AIC-based model selection.
#   - time_to_t0_rev coding documented.
#   - Structured results CSV.

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

monthly = pd.read_csv(PROCESSED / 'repository_month_metrics_with_controls.csv')

monthly['is_abandoned']     = (monthly['cohort'] == 'abandoned').astype(int)
# time_to_t0_rev: 1 = month 24 (furthest from T0), OBS_MONTHS = month 1 (closest).
# Positive interaction coeff → metric increases as T0 approaches for abandoned repos.
monthly['time_to_t0_rev']   = OBS_MONTHS - monthly['months_before_t0'] + 1
monthly['log_commits']      = np.log1p(monthly['monthly_commits'])
monthly['log_total_issues'] = np.log1p(monthly['total_issues'])

# FIX 2: define which outcomes are "issue-count-derived" (endogenous w.r.t.
# log_total_issues) and which are quality/responsiveness outcomes where
# log_total_issues is a legitimate size control.
ISSUE_COUNT_OUTCOMES = {
    'requirement_issues', 'requirement_ratio', 'bug_expectation_issues'
}

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

results_rows = []

for metric in OUTCOMES:
    if metric not in monthly.columns:
        print(f'[SKIP] {metric} not found.')
        continue

    # FIX 2: selective controls — exclude log_total_issues for issue-count outcomes
    if metric in ISSUE_COUNT_OUTCOMES:
        controls    = '+ log_commits'
        control_note = 'log_commits only (log_total_issues excluded: endogenous)'
    else:
        controls    = '+ log_commits + log_total_issues'
        control_note = 'log_commits + log_total_issues'

    dfm = monthly[['repo','is_abandoned','time_to_t0_rev',
                   'log_commits','log_total_issues', metric]
                 ].replace([np.inf, -np.inf], np.nan).dropna()

    if len(dfm) < 50:
        print(f'[SKIP] {metric}: {len(dfm)} obs.')
        continue

    n_repos  = dfm['repo'].nunique()
    formula  = f'{metric} ~ is_abandoned * time_to_t0_rev {controls}'

    print(f'\n{"="*60}')
    print(f'OUTCOME: {metric} | n={len(dfm)} | repos={n_repos}')
    print(f'Controls: {control_note}')
    print(f'Time: time_to_t0_rev=1→month 24 (far); ={OBS_MONTHS}→month 1 (near T0)')

    # Model A: random intercepts
    try:
        fit_a = smf.mixedlm(formula, data=dfm, groups=dfm['repo']).fit(
            reml=False, method='lbfgs')
    except Exception as e:
        print(f'  Model A failed: {e}'); continue

    # Model B: random intercepts + random slopes
    try:
        fit_b = smf.mixedlm(formula, data=dfm, groups=dfm['repo'],
                             re_formula='~time_to_t0_rev').fit(
            reml=False, method='lbfgs')
        use_slopes = True
    except Exception as e:
        print(f'  Random slopes failed ({e}); using intercepts only.')
        fit_b, use_slopes = fit_a, False

    best     = fit_b if (use_slopes and fit_b.aic < fit_a.aic) else fit_a
    mtype    = 'random_slopes' if best is fit_b and use_slopes else 'random_intercepts'
    print(f'  Selected: {mtype} | AIC_A={fit_a.aic:.1f} | AIC_B={fit_b.aic:.1f}')
    print(best.summary())

    with open(RESULTS / f'model_{metric}.txt', 'w') as f:
        f.write(f'Outcome: {metric}\nControls: {control_note}\n'
                f'Model: {mtype}\n'
                f'time_to_t0_rev=1→month 24; ={OBS_MONTHS}→month 1 (near T0)\n')
        f.write(str(best.summary()))

    # FIX 3: repository-level Cohen's d (early vs late window)
    # Aggregate to one value per repo per window half before computing d.
    # This respects the clustered structure and avoids inflating effect sizes.
    early_mb = OBS_MONTHS // 2 + 1   # months further from T0
    late_mb  = OBS_MONTHS // 2       # months closer to T0

    repo_means = dfm.copy()
    repo_means['window'] = np.where(
        repo_means['time_to_t0_rev'] <= late_mb, 'late', 'early')

    def repo_level_cohens_d(window_label):
        sub = (repo_means[repo_means['window'] == window_label]
               .groupby(['repo','is_abandoned'])[metric].mean().reset_index())
        ab = sub[sub['is_abandoned']==1][metric].dropna()
        ac = sub[sub['is_abandoned']==0][metric].dropna()
        if len(ab) < 2 or len(ac) < 2:
            return np.nan
        pooled = np.sqrt((ab.std()**2 + ac.std()**2) / 2)
        return (ab.mean() - ac.mean()) / pooled if pooled > 0 else np.nan

    d_early = repo_level_cohens_d('early')
    d_late  = repo_level_cohens_d('late')

    coef_int = best.params.get('is_abandoned:time_to_t0_rev', np.nan)
    p_int    = best.pvalues.get('is_abandoned:time_to_t0_rev', np.nan)
    coef_ab  = best.params.get('is_abandoned', np.nan)
    p_ab     = best.pvalues.get('is_abandoned', np.nan)

    results_rows.append({
        'outcome':              metric,
        'controls':             control_note,
        'model_type':           mtype,
        'n_observations':       len(dfm),
        'n_repositories':       n_repos,
        'coef_is_abandoned':    round(coef_ab,  4),
        'p_is_abandoned':       round(p_ab,     4),
        'coef_interaction':     round(coef_int, 4),
        'p_interaction_raw':    round(p_int,    4),   # uncorrected
        'p_interaction_fdr':    np.nan,                # filled after loop
        'sig_fdr_q05':          False,                 # filled after loop
        'cohens_d_early':       round(d_early,  3),
        'cohens_d_late':        round(d_late,   3),
        'aic_intercepts':       round(fit_a.aic, 2),
        'aic_slopes':           round(fit_b.aic, 2),
    })

# ── FIX 1: BH-FDR multiple testing correction ─────────────────────────────────
results_df = pd.DataFrame(results_rows)

if len(results_df) > 0:
    raw_ps = results_df['p_interaction_raw'].values
    reject, p_adj, _, _ = multipletests(raw_ps, alpha=0.05, method='fdr_bh')
    results_df['p_interaction_fdr'] = p_adj.round(4)
    results_df['sig_fdr_q05']       = reject

    print('\n' + '='*60)
    print('BH-FDR correction applied across all interaction-term p-values.')
    print('Primary significance criterion: p_interaction_fdr < 0.05')
    print('='*60)

save_csv_checkpoint(results_df, RESULTS / 'mixed_effects_results.csv',
                    '13_mixed_effects_results')

print('\nMODEL RESULTS SUMMARY')
display(results_df[['outcome','controls','model_type','n_observations',
                     'coef_interaction','p_interaction_raw','p_interaction_fdr',
                     'sig_fdr_q05','cohens_d_early','cohens_d_late']])

sig_fdr = results_df[results_df['sig_fdr_q05']]['outcome'].tolist()
sig_raw = results_df[results_df['p_interaction_raw'] < 0.05]['outcome'].tolist()
print(f'\nSignificant (FDR q<.05):  {sig_fdr}')
print(f'Significant (raw p<.05):  {sig_raw}')
print(f'Outcomes downgraded by FDR: {[o for o in sig_raw if o not in sig_fdr]}')
