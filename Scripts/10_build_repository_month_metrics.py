# 10_build_repository_month_metrics.py
# Purpose:
#   Aggregate issue-level data into repository-month metrics.
#   Repository-month is the primary unit of analysis.
#
# No logic changes from previous version.
# Retained fixes: skeleton-first join, strftime month key, NaN sanity check,
# bug_expectation tracked separately, discussion_depth included.

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

issues     = pd.read_csv(PROCESSED / 'issues_with_negotiation_metrics.csv')
repo_panel = pd.read_csv(PROCESSED / 'repo_panel.csv')

for c in ['created_at', 'closed_at', 't0']:
    issues[c] = pd.to_datetime(issues[c], utc=True, errors='coerce')
for c in ['t0', 'window_start', 'window_end']:
    repo_panel[c] = pd.to_datetime(repo_panel[c], utc=True, errors='coerce')

# ── Skeleton ───────────────────────────────────────────────────────────────────
skeleton_rows = []
for _, r in repo_panel.iterrows():
    t0 = r['t0']
    for mb in range(1, OBS_MONTHS + 1):
        skeleton_rows.append({
            'repo':             r['full_name'],
            'pair_id':          r['pair_id'],
            'cohort':           r['cohort'],
            'language':         r['language'],
            'domain':           r.get('domain', 'other'),
            'poor_match_flag':  r.get('poor_match_flag', False),
            'month':            (t0 - relativedelta(months=mb)).strftime('%Y-%m'),
            'months_before_t0': mb,
        })
skeleton = pd.DataFrame(skeleton_rows)

# ── Month key for issues ───────────────────────────────────────────────────────
issues['months_before_t0'] = (
    (issues['t0'].dt.year  - issues['created_at'].dt.year)  * 12 +
    (issues['t0'].dt.month - issues['created_at'].dt.month)
)
issues['month'] = issues['created_at'].dt.strftime('%Y-%m')

issues_w = issues[(issues['months_before_t0'] >= 1) &
                   (issues['months_before_t0'] <= OBS_MONTHS)].copy()
req      = issues_w[issues_w['final_is_requirement'] == 1].copy()

# ── All-issues aggregation ─────────────────────────────────────────────────────
GRPKEYS = ['repo','pair_id','cohort','language','domain','month','months_before_t0']

all_monthly = issues_w.groupby(GRPKEYS, as_index=False).agg(
    total_issues=  ('issue_id', 'count'),
    total_comments=('comments_count', 'sum'),
)

# ── Requirement aggregation ────────────────────────────────────────────────────
def safe_mean(x):
    v = x.dropna()
    return v.mean() if len(v) else np.nan

def ignored_ratio(x):
    return 1 - np.nanmean(x) if len(x) else np.nan

req_monthly = req.groupby(GRPKEYS, as_index=False).agg(
    requirement_issues=       ('issue_id',                'count'),
    bug_expectation_issues=   ('final_req_type',
                               lambda x: (x=='bug_expectation').sum()),
    avg_req_length_words=     ('text_len_words',          safe_mean),
    avg_actionability_score=  ('actionability_score',     safe_mean),
    avg_vagueness_density=    ('vagueness_density',       safe_mean),
    rationale_ratio=          ('has_rationale',           'mean'),
    acceptance_ratio=         ('has_acceptance',          'mean'),
    req_closure_rate=         ('is_closed',               'mean'),
    ignored_requirement_ratio=('has_non_author_response', ignored_ratio),
    avg_first_response_hours= ('first_response_hours',    safe_mean),
    avg_unique_commenters=    ('unique_commenters',       safe_mean),
    avg_discussion_depth=     ('discussion_depth',        safe_mean),
    roadmap_issue_ratio=      ('milestone', lambda x: x.notna().mean()),
    milestone_present_any=    ('milestone', lambda x: int(x.notna().any())),
)

monthly = (skeleton
           .merge(all_monthly, on=GRPKEYS, how='left')
           .merge(req_monthly,  on=GRPKEYS, how='left'))

for c in ['total_issues','total_comments','requirement_issues',
          'bug_expectation_issues','milestone_present_any']:
    monthly[c] = monthly[c].fillna(0)

monthly['requirement_ratio'] = (
    monthly['requirement_issues'] /
    monthly['total_issues'].replace(0, np.nan)
)

# ── Sanity check ───────────────────────────────────────────────────────────────
n     = len(monthly)
n_nan = monthly['requirement_issues'].isna().sum()
print(f'Panel rows: {n} | NaN requirement_issues: {n_nan} (should be 0)')
if n_nan > 0:
    print('[WARNING] NaN in requirement_issues — check month-key consistency.')

save_csv_checkpoint(monthly, PROCESSED / 'repository_month_metrics.csv',
                    '10_repository_month_metrics')
display(monthly.head())
