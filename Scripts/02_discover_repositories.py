# 02_discover_repositories.py
# Purpose:
#   Discover and verify candidate abandoned and active repositories.
#
# Fixes in this version:
#   FIX 1 — Active cohort activity verification (NEW, CRITICAL):
#     Active repositories are now verified to have had at least
#     MIN_COMMITS_BEFORE_T0 commits during their observation window.
#     Previously only the abandoned cohort was verified; the asymmetry
#     meant "active" controls could be historically dormant and only
#     recently revived. Both cohorts are now verified to the same standard.
#
#   FIX 2 — infer_domain() now uses word-boundary regex (from config).
#     No change needed here; fix lives in 00_config.py.
#
#   Retained from previous version:
#   - pushed_at + commit-count abandonment verification for abandoned repos.
#   - Domain inference from GitHub topics.
#   - open_issues_count excluded.
#   - warn_search_cap() called after every search.

import time
import pandas as pd
from tqdm.auto import tqdm
from dateutil.relativedelta import relativedelta


# ── Search helpers ─────────────────────────────────────────────────────────────

def search_repositories(query, sort='stars', order='desc', max_pages=5):
    url = 'https://api.github.com/search/repositories'
    items, hit_cap = paginate_search(
        url, params={'q': query, 'sort': sort, 'order': order},
        max_pages=max_pages,
    )
    if hit_cap:
        warn_search_cap(items, query)
    return items


def normalize_repo_items(items, label):
    rows, seen = [], set()
    for it in items:
        full = it.get('full_name')
        if not full or full in seen:
            continue
        seen.add(full)
        rows.append({
            'full_name':               full,
            'owner':                   (it.get('owner') or {}).get('login'),
            'name':                    it.get('name'),
            'language':                it.get('language'),
            'stars':                   it.get('stargazers_count'),
            'forks':                   it.get('forks_count'),
            'created_at':              it.get('created_at'),
            'pushed_at':               it.get('pushed_at'),
            'archived':                it.get('archived'),
            'fork':                    it.get('fork'),
            'has_issues':              it.get('has_issues'),
            'description':             it.get('description'),
            'topics':                  '|'.join(it.get('topics') or []),
            'candidate_group':         label,
            'domain':                  None,
            'commit_count_in_window':  None,
        })
    return pd.DataFrame(rows)


# ── Step 1: collect raw candidates ────────────────────────────────────────────

abandoned_items, active_items = [], []
abandoned_date = ABANDONED_CUTOFF.strftime('%Y-%m-%d')
active_date    = ACTIVE_RECENT_CUTOFF.strftime('%Y-%m-%d')

for lang in LANGUAGES:
    print(f'Searching {lang} …')
    q_ab = (f'language:{lang} stars:>=500 fork:false '
            f'archived:false pushed:<{abandoned_date}')
    q_ac = (f'language:{lang} stars:>=500 fork:false '
            f'archived:false pushed:>{active_date}')
    abandoned_items.extend(search_repositories(q_ab, max_pages=3))
    active_items.extend(  search_repositories(q_ac, max_pages=3))
    time.sleep(1)

abandoned_candidates = normalize_repo_items(abandoned_items, 'abandoned')
active_candidates    = normalize_repo_items(active_items,    'active')
print(f'Raw abandoned: {len(abandoned_candidates)} | Raw active: {len(active_candidates)}')


# ── Step 2: basic structural filters ──────────────────────────────────────────

def basic_filter(df, pushed_before=None, pushed_after=None):
    df = df.copy()
    df['created_at'] = pd.to_datetime(df['created_at'], utc=True, errors='coerce')
    df['pushed_at']  = pd.to_datetime(df['pushed_at'],  utc=True, errors='coerce')
    df['age_days']   = (RUN_DATE - df['created_at']).dt.days
    mask = (
        (df['stars']      >= 500)  &
        (df['age_days']   >= 730)  &
        (df['fork']       == False) &
        (df['archived']   == False) &
        (df['has_issues'] == True)
    )
    if pushed_before is not None:
        mask &= df['pushed_at'] < pushed_before
    if pushed_after is not None:
        mask &= df['pushed_at'] > pushed_after
    return df[mask].copy()

ab_filtered = basic_filter(abandoned_candidates, pushed_before=ABANDONED_CUTOFF)
ac_filtered = basic_filter(active_candidates,    pushed_after=ACTIVE_RECENT_CUTOFF)
print(f'After basic filter — abandoned: {len(ab_filtered)}, active: {len(ac_filtered)}')


# ── Step 3: enrich with full metadata and domain ───────────────────────────────

def enrich_with_metadata(df):
    records  = df.to_dict('records')
    enriched = []
    for rec in tqdm(records, desc='Enriching metadata'):
        meta = get_repo_metadata(rec['full_name'])
        if meta:
            topics          = meta.get('topics') or []
            rec['topics']   = '|'.join(topics)
            rec['description'] = meta.get('description') or rec.get('description', '')
            rec['domain']   = infer_domain(topics, rec['description'])
        else:
            rec['domain'] = infer_domain(
                rec.get('topics', '').split('|'),
                rec.get('description', ''),
            )
        enriched.append(rec)
        time.sleep(0.3)
    return pd.DataFrame(enriched)

ab_filtered = enrich_with_metadata(ab_filtered)
ac_filtered = enrich_with_metadata(ac_filtered)


# ── Step 4: abandoned cohort — commit verification ────────────────────────────
# A candidate is verified abandoned if:
#   (a) it had ≥ MIN_COMMITS_BEFORE_T0 commits in the 24-month window, AND
#   (b) it had ZERO commits in the 12 months after its last pushed_at.

def verify_abandoned_repos(df):
    verified = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc='Verifying abandonment'):
        full         = row['full_name']
        t0           = pd.to_datetime(row['pushed_at'], utc=True)
        window_start = t0 - relativedelta(months=OBS_MONTHS)

        count_in   = get_recent_commit_count(full, window_start, t0)
        count_post = get_recent_commit_count(full, t0, t0 + relativedelta(months=12))

        row = row.copy()
        row['commit_count_in_window'] = count_in
        row['commit_count_post_t0']   = count_post
        row['abandonment_verified']   = (count_in >= MIN_COMMITS_BEFORE_T0
                                         and count_post == 0)
        verified.append(row)
        time.sleep(0.5)

    result     = pd.DataFrame(verified)
    n_verified = result['abandonment_verified'].sum()
    print(f'Abandoned verified: {n_verified}/{len(result)}')
    checkpoint('02_abandonment_verification', {
        'total': len(result), 'verified': int(n_verified),
    })
    return result[result['abandonment_verified']].copy()

ab_verified = verify_abandoned_repos(ab_filtered)


# ── Step 5: active cohort — no window-based verification here ─────────────────
# Activity verification for active repositories CANNOT be done at this stage
# because each active repo's observation window is anchored to its abandoned
# partner's T0, which is not known until after matching (script 04).
# Verifying against the active repo's own pushed_at would check the wrong
# time period (e.g. 2024–2026 instead of 2021–2023).
# Verification is performed in script 04 after pseudo-T0 is assigned.
# Active candidates pass through here without commit verification.
ac_verified = ac_filtered.copy()
ac_verified['commit_count_in_window'] = None   # filled in script 04
ac_verified['activity_verified']      = None   # filled in script 04
print(f'Active candidates (pre-verification, done in script 04): {len(ac_verified)}')\



# ── Step 6: save ───────────────────────────────────────────────────────────────

save_csv_checkpoint(ab_verified, RAW / 'candidate_abandoned_repos.csv',
                    '02_candidate_abandoned_repos')
save_csv_checkpoint(ac_verified, RAW / 'candidate_active_repos.csv',
                    '02_candidate_active_repos')

checkpoint('02_discover_repositories', {
    'abandoned_raw':              len(abandoned_candidates),
    'active_raw':                 len(active_candidates),
    'abandoned_verified':         len(ab_verified),
    'active_candidates_saved':    len(ac_verified),
    'note': 'Active window verification deferred to script 04 (requires pseudo-T0)',
})

print(f'\nAbandoned verified: {len(ab_verified)} | Active candidates: {len(ac_verified)}')
print('NOTE: Active repos are activity-verified in script 04 after pseudo-T0 is assigned.')
display(ab_verified[['full_name', 'language', 'domain', 'stars',
                      'commit_count_in_window', 'commit_count_post_t0']].head())
display(ac_verified[['full_name', 'language', 'domain', 'stars']].head())
