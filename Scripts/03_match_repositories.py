# 03_match_repositories.py
# Purpose:
#   Match abandoned repositories to active controls using nearest-neighbour
#   matching within (language × domain) strata.
#
# No logic changes from previous version.
# Retained fixes: domain stratification, balance table, poor-match flag,
# open_issues_count excluded, T0 sourced from abandoned_t0.

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy import stats

abandoned_candidates = pd.read_csv(RAW / 'candidate_abandoned_repos.csv')
active_candidates    = pd.read_csv(RAW / 'candidate_active_repos.csv')


def prepare_frame(df):
    df = df.copy()
    for c in ['created_at', 'pushed_at']:
        df[c] = pd.to_datetime(df[c], utc=True, errors='coerce')
    df['age_days']  = (RUN_DATE - df['created_at']).dt.days
    df['log_stars'] = np.log1p(df['stars'].fillna(0))
    df['log_forks'] = np.log1p(df['forks'].fillna(0))
    df['domain']    = df['domain'].fillna('other')
    return df

ab = prepare_frame(abandoned_candidates)
ac = prepare_frame(active_candidates)

ab_f = ab[(ab['stars'] >= 500) & (ab['age_days'] >= 730) &
          (ab['fork'] == False) & (ab['archived'] == False)].copy()
ac_f = ac[(ac['stars'] >= 500) & (ac['age_days'] >= 730) &
          (ac['fork'] == False) & (ac['archived'] == False)].copy()

print(f'Eligible abandoned: {len(ab_f)} | Eligible active: {len(ac_f)}')

MATCH_FEATURES = ['log_stars', 'log_forks', 'age_days']


def match_repos(ab_f, ac_f, n_pairs):
    """
    1-to-1 nearest-neighbour matching within (language × domain) strata.
    Falls back to language-only pool if domain stratum is empty.
    Flags pairs with distance > MATCH_DISTANCE_THRESHOLD for robustness check 4.
    """
    pairs = []
    used  = set()

    ab_f = ab_f.copy()
    ac_f = ac_f.copy()
    ab_f['stratum'] = ab_f['language'].fillna('Unknown') + '::' + ab_f['domain']
    ac_f['stratum'] = ac_f['language'].fillna('Unknown') + '::' + ac_f['domain']

    for stratum, ab_stratum in ab_f.groupby('stratum'):
        lang, domain = stratum.split('::', 1)
        ac_stratum   = ac_f[ac_f['stratum'] == stratum].copy()

        if len(ac_stratum) == 0:
            ac_stratum = ac_f[ac_f['language'] == lang].copy()
            if len(ac_stratum) == 0:
                print(f'  No controls for {stratum} — skipping.')
                continue
            print(f'  Domain fallback for {stratum}: using language-only pool.')

        combined = pd.concat([ab_stratum[MATCH_FEATURES],
                              ac_stratum[MATCH_FEATURES]], axis=0).fillna(0)
        scaler   = StandardScaler().fit(combined)
        Xc       = scaler.transform(ac_stratum[MATCH_FEATURES].fillna(0))
        nn       = NearestNeighbors(n_neighbors=min(10, len(ac_stratum))).fit(Xc)

        for _, row in ab_stratum.sort_values('stars', ascending=False).iterrows():
            if len(pairs) >= n_pairs:
                break
            xa = scaler.transform(
                pd.DataFrame([row[MATCH_FEATURES].fillna(0).to_dict()])
            )
            distances, indices = nn.kneighbors(xa)
            chosen = chosen_dist = None
            for d, i in zip(distances[0], indices[0]):
                cand = ac_stratum.iloc[i]
                if cand['full_name'] not in used:
                    chosen, chosen_dist = cand, float(d)
                    break
            if chosen is None:
                print(f'  No unused control for {row["full_name"]} — skipping.')
                continue

            pairs.append({
                'pair_id':             len(pairs) + 1,
                'abandoned_repo':      row['full_name'],
                'active_repo':         chosen['full_name'],
                'language':            lang,
                'domain':              domain,
                'stratum':             stratum,
                'match_distance':      chosen_dist,
                'poor_match_flag':     chosen_dist > MATCH_DISTANCE_THRESHOLD,
                'abandoned_t0':        row['pushed_at'],
                'active_pushed_at':    chosen['pushed_at'],
                'abandoned_stars':     row['stars'],
                'active_stars':        chosen['stars'],
                'abandoned_log_stars': row['log_stars'],
                'active_log_stars':    chosen['log_stars'],
                'abandoned_log_forks': row['log_forks'],
                'active_log_forks':    chosen['log_forks'],
                'abandoned_age_days':  row['age_days'],
                'active_age_days':     chosen['age_days'],
            })
            used.add(chosen['full_name'])
        if len(pairs) >= n_pairs:
            break

    return pd.DataFrame(pairs)


matched = match_repos(ab_f, ac_f, N_ABANDONED)
print(f'Matched pairs: {len(matched)} | Poor-match: {matched["poor_match_flag"].sum()}')


# ── Covariate balance table ────────────────────────────────────────────────────

def smd(a, b):
    pooled = np.sqrt((np.std(a, ddof=1)**2 + np.std(b, ddof=1)**2) / 2)
    return (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else 0.0

balance_rows = []
for feat, ac, bc in [('log_stars', 'abandoned_log_stars', 'active_log_stars'),
                      ('log_forks', 'abandoned_log_forks', 'active_log_forks'),
                      ('age_days',  'abandoned_age_days',  'active_age_days')]:
    av, bv = matched[ac].dropna(), matched[bc].dropna()
    _, p   = stats.ttest_ind(av, bv, equal_var=False)
    s      = smd(av, bv)
    balance_rows.append({
        'variable':              feat,
        'abandoned_mean':        round(av.mean(), 3),
        'active_mean':           round(bv.mean(), 3),
        'abandoned_sd':          round(av.std(ddof=1), 3),
        'active_sd':             round(bv.std(ddof=1), 3),
        'standardised_mean_diff': round(s, 3),
        'ttest_p':               round(p, 4),
        'balance_ok':            abs(s) < 0.25,
    })

balance_table = pd.DataFrame(balance_rows)
display(balance_table)

poor = balance_table[~balance_table['balance_ok']]
if len(poor):
    print(f'[WARNING] Poor balance: {list(poor["variable"])}')

save_csv_checkpoint(balance_table, RESULTS / 'covariate_balance_table.csv',
                    '03_covariate_balance_table')
save_csv_checkpoint(matched, PROCESSED / 'matched_repositories.csv',
                    '03_matched_repositories')

checkpoint('03_match_repositories', {
    'pairs':        len(matched),
    'poor_match':   int(matched['poor_match_flag'].sum()),
    'max_smd':      float(balance_table['standardised_mean_diff'].abs().max()),
})
display(matched[['pair_id','abandoned_repo','active_repo',
                 'language','domain','match_distance','poor_match_flag']].head(10))
