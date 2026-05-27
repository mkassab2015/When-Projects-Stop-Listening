# 04_build_repo_panel.py
# Purpose:
#   Build the 24-month observation panel and verify active cohort activity
#   on the correct temporal window.
#
# Fix applied — active cohort window verification (SERIOUS):
#   Active repositories are now verified against their actual pseudo-T0
#   observation window (the abandoned partner's T0 window), not against
#   their own pushed_at window. Pairs where the active repo had fewer than
#   MIN_COMMITS_BEFORE_T0 commits in the pseudo-T0 window are dropped and
#   logged. Dropped pairs are saved for manuscript transparency.

import time
import pandas as pd
from tqdm.auto import tqdm
from dateutil.relativedelta import relativedelta

matched  = pd.read_csv(PROCESSED / 'matched_repositories.csv')

# ── Build panel rows with active-repo window verification ─────────────────────

kept_rows    = []
dropped_pairs = []

for _, r in tqdm(matched.iterrows(), total=len(matched),
                 desc='Building panel + verifying active windows'):

    t0           = pd.to_datetime(r['abandoned_t0'], utc=True)
    window_start = t0 - relativedelta(months=OBS_MONTHS)
    window_end   = t0

    active_repo = r['active_repo']

    # Verify active repo had sufficient activity in the PSEUDO-T0 window.
    # This is the window the active repo will actually be analysed in.
    # Using any other window (e.g. the repo's own pushed_at window) would
    # check the wrong time period and allow dormant-then-revived repos through.
    count_in = get_recent_commit_count(active_repo, window_start, window_end)
    time.sleep(0.3)

    if count_in < MIN_COMMITS_BEFORE_T0:
        print(f'  [DROP] pair {r["pair_id"]}: {active_repo} had {count_in} commits '
              f'in pseudo-T0 window ({window_start.date()}–{window_end.date()}).')
        dropped_pairs.append({
            'pair_id':                r['pair_id'],
            'abandoned_repo':         r['abandoned_repo'],
            'active_repo':            active_repo,
            'reason':                 'active_insufficient_activity_in_pseudo_T0_window',
            'commit_count_in_window': count_in,
            'window_start':           str(window_start.date()),
            'window_end':             str(window_end.date()),
        })
        continue  # drop this pair entirely

    # Both repos pass — add both to the panel
    for cohort, full_name in [('abandoned', r['abandoned_repo']),
                               ('active',    active_repo)]:
        kept_rows.append({
            'pair_id':         r['pair_id'],
            'full_name':       full_name,
            'cohort':          cohort,
            'language':        r['language'],
            'domain':          r['domain'],
            'stratum':         r.get('stratum', f"{r['language']}::{r['domain']}"),
            'poor_match_flag': r['poor_match_flag'],
            't0':              t0,
            'window_start':    window_start,
            'window_end':      window_end,
        })

repo_panel = pd.DataFrame(kept_rows)
dropped_df = pd.DataFrame(dropped_pairs)

n_original = len(matched)
n_dropped  = len(dropped_pairs)
n_kept     = len(repo_panel) // 2

print(f'\nOriginal matched pairs:  {n_original}')
print(f'Dropped (active repo failed pseudo-T0 activity check): {n_dropped}')
print(f'Final pairs in panel:    {n_kept}')

if n_dropped > 0:
    print(f'[WARNING] {n_dropped} pairs dropped. Report in manuscript.')
    save_csv_checkpoint(dropped_df, PROCESSED / 'dropped_pairs_activity_check.csv')

if n_kept < N_ABANDONED * 0.8:
    print(f'[WARNING] <80% of pairs retained. Consider relaxing '
          f'MIN_COMMITS_BEFORE_T0 or enlarging candidate pool.')

save_csv_checkpoint(repo_panel, PROCESSED / 'repo_panel.csv', '04_repo_panel')

checkpoint('04_build_repo_panel', {
    'original_pairs':        n_original,
    'dropped_pairs':         n_dropped,
    'final_pairs':           n_kept,
    'panel_rows':            len(repo_panel),
    'MIN_COMMITS_BEFORE_T0': MIN_COMMITS_BEFORE_T0,
    'note': 'Active repos verified in pseudo-T0 window (correct temporal scope).',
})

print(f'\nPanel shape: {repo_panel.shape}')
print(repo_panel['cohort'].value_counts())
print()
print('Threats to validity:')
print('  - Active repos now verified for activity in their actual analysis window.')
print('  - Calendar-time confound (pseudo-T0) mitigated by monthly_commits control.')
print('  - Dropped pairs reduce N; report in manuscript with reasons.')
display(repo_panel.head(6))
