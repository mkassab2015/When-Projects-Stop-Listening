# 09_compute_negotiation_metrics.py
# Purpose:
#   Compute stakeholder negotiation and responsiveness metrics at issue level.
#
# Fixes in this version:
#   FIX 1 — discussion_depth floor effect (SERIOUS):
#     Previously returned min=1 even for single-comment threads, conflating
#     "had any response" with "had negotiation". Now:
#       - Issues with ZERO non-author comments → discussion_depth = 0.
#       - Issues with ≥1 non-author comment → turns = number of speaker
#         role alternations (min 1 for a single-comment thread).
#     This makes the metric genuinely zero for ignored issues, which is
#     substantively meaningful and avoids near-duplication with
#     ignored_requirement_ratio.
#
#   FIX 2 — first_response_hours outlier winsorisation (MINOR):
#     Winsorise at the 99th percentile before saving to prevent extreme
#     values (issues that receive a response months/years later) from
#     distorting monthly means. The raw value is also retained as
#     first_response_hours_raw for transparency.

import pandas as pd
import numpy as np

issues        = pd.read_csv(PROCESSED / 'issues_final_classified.csv')
comments_path = RAW / 'comments_raw.csv'

for c in ['created_at', 'closed_at', 't0']:
    issues[c] = pd.to_datetime(issues[c], utc=True, errors='coerce')

# ── Negotiation metrics from comments ─────────────────────────────────────────

if comments_path.exists() and comments_path.stat().st_size > 0:
    comments = pd.read_csv(comments_path)
    comments['comment_created_at'] = pd.to_datetime(
        comments['comment_created_at'], utc=True, errors='coerce')

    ca = comments.merge(
        issues[['issue_id', 'author', 'created_at']], on='issue_id', how='left')
    ca['is_non_author'] = (ca['comment_author'].notna() &
                           (ca['comment_author'] != ca['author']))

    # First non-author response
    first_resp = (
        ca[ca['is_non_author']]
        .groupby('issue_id')['comment_created_at'].min()
        .reset_index().rename(columns={'comment_created_at': 'first_non_author_at'})
    )

    # Unique commenters
    unique_comm = (
        ca.groupby('issue_id')['comment_author'].nunique()
        .reset_index().rename(columns={'comment_author': 'unique_commenters'})
    )

    # FIX: discussion_depth = 0 for issues with no non-author response.
    # For issues with ≥1 non-author comment, count speaker role alternations.
    # This ensures ignored issues score 0 (not 1), making the metric
    # substantively different from and complementary to ignored_requirement_ratio.
    def discussion_depth(grp):
        grp = grp.sort_values('comment_created_at')
        non_author_count = grp['is_non_author'].sum()
        if non_author_count == 0:
            return 0   # FIX: was 1 previously — now correctly 0
        roles = grp['is_non_author'].tolist()
        turns = 1 + sum(1 for a, b in zip(roles[:-1], roles[1:]) if a != b)
        return turns

    depth = (
        ca.groupby('issue_id').apply(discussion_depth, include_groups=False)
        .reset_index().rename(columns={0: 'discussion_depth'})
    )

    issues = (issues
              .merge(first_resp,  on='issue_id', how='left')
              .merge(unique_comm, on='issue_id', how='left')
              .merge(depth,       on='issue_id', how='left'))
else:
    print('[WARNING] comments_raw.csv missing — negotiation metrics will be NaN.')
    issues['first_non_author_at'] = pd.NaT
    issues['unique_commenters']   = np.nan
    issues['discussion_depth']    = np.nan

# ── Derived metrics ────────────────────────────────────────────────────────────

issues['first_non_author_at']     = pd.to_datetime(
    issues['first_non_author_at'], utc=True, errors='coerce')
issues['has_non_author_response'] = issues['first_non_author_at'].notna().astype(int)

# Raw response time (NaN for ignored issues — intentional)
issues['first_response_hours_raw'] = (
    (issues['first_non_author_at'] - issues['created_at'])
    .dt.total_seconds() / 3600
)

# FIX: Winsorise response time using a theoretically motivated ceiling.
# We use 8,760 hours (= 365 days = 1 full year) as the cap. This is grounded
# in the observation window length: any response arriving > 1 year after the
# issue was opened is functionally equivalent to never receiving a response
# within a meaningful timeframe for requirements work. Using a fixed theoretical
# cap avoids cohort-specific thresholds that would distort cross-cohort comparisons
# (a pooled 99th-pct threshold is partly driven by the abandoned cohort's extreme
# late responses, potentially compressing the active cohort's distribution unfairly).
# The raw value is preserved in first_response_hours_raw for transparency.
RESPONSE_HOURS_CAP = 8760   # 1 year in hours — matches the OBS_MONTHS window
issues['first_response_hours'] = issues['first_response_hours_raw'].clip(upper=RESPONSE_HOURS_CAP)
n_clipped = (issues['first_response_hours_raw'] > RESPONSE_HOURS_CAP).sum()
print(f'first_response_hours capped at {RESPONSE_HOURS_CAP}h (1 year): '
      f'{n_clipped} issues clipped ({n_clipped/len(issues):.1%})')

issues['unique_commenters']  = issues['unique_commenters'].fillna(0)
issues['discussion_depth']   = issues['discussion_depth'].fillna(0)
issues['time_to_close_days'] = (
    (issues['closed_at'] - issues['created_at']).dt.total_seconds() / 86400)
issues['is_closed']          = issues['closed_at'].notna().astype(int)

save_csv_checkpoint(issues, PROCESSED / 'issues_with_negotiation_metrics.csv',
                    '09_issues_with_negotiation_metrics')

print(f'Response rate: {issues["has_non_author_response"].mean():.1%}')
print(f'Median response (hours, winsorised): {issues["first_response_hours"].median():.1f}')
print(f'discussion_depth == 0: {(issues["discussion_depth"]==0).mean():.1%}')
print(f'discussion_depth == 1: {(issues["discussion_depth"]==1).mean():.1%}')
print(f'discussion_depth >= 2: {(issues["discussion_depth"]>=2).mean():.1%}')
display(issues[['title','final_is_requirement','has_non_author_response',
                'first_response_hours','unique_commenters','discussion_depth']].head())
