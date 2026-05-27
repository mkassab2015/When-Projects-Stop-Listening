# 12_generate_plots.py
# Purpose:
#   Generate manuscript-quality trend plots for all decay metrics.
#
# Fix in this version:
#   FIX — Bootstrap CI seeded with np.random.seed(42) at the top of the script
#     so that confidence band positions are exactly reproducible across reruns.
#     Previously the CIs varied slightly each run, making figures non-reproducible.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats

np.random.seed(42)   # FIX: seed for reproducible bootstrap CIs

sns.set_theme(style='whitegrid', font_scale=1.1)
PALETTE    = {'abandoned': '#d62728', 'active': '#1f77b4'}
ALPHA_BAND = 0.20

monthly = pd.read_csv(PROCESSED / 'repository_month_metrics_with_controls.csv')


def bootstrap_ci(values, n_boot=500, ci=95):
    values = np.array(values.dropna())
    if len(values) == 0:
        return np.nan, np.nan
    boots = np.random.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return np.percentile(boots, (100-ci)/2), np.percentile(boots, 100-(100-ci)/2)


def compute_trend(df, metric, exclude_poor_match=False):
    if exclude_poor_match and 'poor_match_flag' in df.columns:
        df = df[~df['poor_match_flag'].astype(bool)]
    rows = []
    for (cohort, mb), grp in df.groupby(['cohort','months_before_t0']):
        vals = grp[metric].dropna()
        lo, hi = bootstrap_ci(vals)
        rows.append({'cohort': cohort, 'months_before_t0': mb,
                     'mean': vals.mean(), 'ci_lo': lo, 'ci_hi': hi, 'n': len(vals)})
    return pd.DataFrame(rows)


def significance_annotation(df, metric):
    annotations = []
    for label, (lo_mb, hi_mb) in [('Early',(19,24)),('Late',(1,6))]:
        sub = df[df['months_before_t0'].between(lo_mb, hi_mb)]
        ab  = sub[sub['cohort']=='abandoned'][metric].dropna()
        ac  = sub[sub['cohort']=='active'][metric].dropna()
        if len(ab) < 3 or len(ac) < 3: continue
        _, p = stats.ttest_ind(ab, ac, equal_var=False)
        annotations.append(f'{label}: {"p<.001" if p<0.001 else f"p={p:.3f}"}')
    return '  |  '.join(annotations)


def plot_metric(ax, df, metric, title, ylabel, exclude_poor_match=False):
    trend = compute_trend(df, metric, exclude_poor_match)
    annot = significance_annotation(df, metric)
    for cohort, sub in trend.groupby('cohort'):
        sub   = sub.sort_values('months_before_t0', ascending=False)
        color = PALETTE.get(cohort, 'grey')
        ax.plot(sub['months_before_t0'], sub['mean'],
                marker='o', markersize=3, color=color, label=cohort.capitalize())
        ax.fill_between(sub['months_before_t0'], sub['ci_lo'], sub['ci_hi'],
                        alpha=ALPHA_BAND, color=color)
    ax.set_xlim(OBS_MONTHS+0.5, 0.5)
    ax.set_xlabel('Months before T₀', fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    if annot:
        ax.text(0.02, 0.97, annot, transform=ax.transAxes, fontsize=7.5,
                va='top', ha='left', color='#444444')


METRIC_SPECS = [
    ('requirement_issues',        'Requirement Activity',        'Req. issues / month'),
    ('requirement_ratio',         'Requirement Ratio',           'Prop. issues that are req.'),
    ('ignored_requirement_ratio', 'Ignored Requirement Ratio',   'Prop. with no response'),
    ('avg_actionability_score',   'Requirement Elaboration',     'Avg actionability (0–4)'),
    ('avg_vagueness_density',     'Vagueness Density',           'Vague words / word'),
    ('roadmap_issue_ratio',       'Roadmap Activity',            'Prop. req. with milestone'),
    ('avg_first_response_hours',  'Response Time (conditional)', 'Avg hours to first response'),
    ('avg_discussion_depth',      'Negotiation Depth',           'Avg discussion turns'),
    ('monthly_commits',           'General Activity (control)',  'Commits / month'),
]

# Individual plots
for col, title, ylabel in METRIC_SPECS:
    if col not in monthly.columns: continue
    fig, ax = plt.subplots(figsize=(8, 4.5))
    plot_metric(ax, monthly, col, title, ylabel)
    plt.tight_layout()
    out = FIGURES / f'{col}_trend.png'
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.show(); plt.close()
    print(f'Saved: {out}')

# Combined four-panel figure
FOUR_PANEL = [
    ('requirement_issues',        'Dimension 1\nRequirement Activity',    'Req. issues / month'),
    ('avg_actionability_score',   'Dimension 2\nRequirement Elaboration', 'Avg actionability (0–4)'),
    ('avg_discussion_depth',      'Dimension 3\nStakeholder Negotiation', 'Avg discussion turns'),
    ('ignored_requirement_ratio', 'Dimension 4\nRequirement Responsiveness','Prop. ignored'),
]

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle('Requirements Decay: Abandoned vs Active Repositories',
             fontsize=13, fontweight='bold', y=1.01)
for ax, (col, title, ylabel) in zip(axes.flat, FOUR_PANEL):
    if col in monthly.columns:
        plot_metric(ax, monthly, col, title, ylabel)
    else:
        ax.set_visible(False)
handles = [mpatches.Patch(color=PALETTE['abandoned'], label='Abandoned'),
           mpatches.Patch(color=PALETTE['active'],    label='Active')]
fig.legend(handles=handles, loc='lower center', ncol=2,
           fontsize=11, bbox_to_anchor=(0.5, -0.04))
plt.tight_layout()
out = FIGURES / 'four_panel_decay_dimensions.png'
plt.savefig(out, dpi=200, bbox_inches='tight')
plt.show(); plt.close()
print(f'Saved combined figure: {out}')

# Sensitivity: well-matched pairs only
if 'poor_match_flag' in monthly.columns and monthly['poor_match_flag'].any():
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f'Sensitivity: Well-Matched Pairs Only (distance ≤ {MATCH_DISTANCE_THRESHOLD})',
                 fontsize=13, fontweight='bold')
    for ax, (col, title, ylabel) in zip(axes.flat, FOUR_PANEL):
        if col in monthly.columns:
            plot_metric(ax, monthly, col, title, ylabel, exclude_poor_match=True)
    fig.legend(handles=handles, loc='lower center', ncol=2,
               fontsize=11, bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout()
    out = FIGURES / 'four_panel_well_matched_only.png'
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.show(); plt.close()
    print(f'Saved sensitivity figure: {out}')
