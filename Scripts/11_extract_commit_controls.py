# 11_extract_commit_controls.py
# Purpose:
#   Extract monthly commit counts as a general-activity control variable.
#   Commit counts are used only as controls, not outcomes.
#   Auto-resume on Colab disconnect preserved.
#
# No logic changes from previous version.

import time
import pandas as pd
from tqdm.auto import tqdm
from dateutil.relativedelta import relativedelta

repo_panel = pd.read_csv(PROCESSED / 'repo_panel.csv')
for c in ['t0','window_start','window_end']:
    repo_panel[c] = pd.to_datetime(repo_panel[c], utc=True, errors='coerce')

commit_path = RAW  / 'commits_raw.csv'
done_path   = CHECKPOINTS / 'commits_done_repos.csv'
rows, done  = [], set()

if commit_path.exists():
    rows = pd.read_csv(commit_path).to_dict('records')
if done_path.exists():
    done = set(pd.read_csv(done_path)['repo'].dropna().unique())


def get_commits_for_repo(full_name, start, end):
    url    = f'https://api.github.com/repos/{full_name}/commits'
    result = []
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + relativedelta(months=1), end)
        for page in range(1, 11):
            r = github_get(url, params={
                'since': cursor.isoformat(), 'until': chunk_end.isoformat(),
                'per_page': 100, 'page': page})
            if r is None: break
            data = r.json()
            if not isinstance(data, list) or not data: break
            for c in data:
                result.append({'repo': full_name, 'sha': c.get('sha'),
                                'commit_date': (c.get('commit') or {})
                                               .get('committer', {}).get('date')})
            if len(data) < 100: break
            if page == 10:
                print(f'  [WARNING] {full_name} may exceed 1000 commits in '
                      f'{cursor.strftime("%Y-%m")}')
        cursor = chunk_end
        time.sleep(0.2)
    return result


for _, r in tqdm(repo_panel.iterrows(), total=len(repo_panel), desc='Commits'):
    full = r['full_name']
    if full in done: continue
    try:
        rows.extend(get_commits_for_repo(
            full, r['window_start'].to_pydatetime(), r['window_end'].to_pydatetime()))
        pd.DataFrame(rows).to_csv(commit_path, index=False)
        done.add(full)
        pd.DataFrame({'repo': sorted(done)}).to_csv(done_path, index=False)
        checkpoint('11_progress', {'last': full, 'done': len(done), 'rows': len(rows)})
        time.sleep(1)
    except Exception as e:
        checkpoint('11_error', {'repo': full, 'error': str(e)})
        raise

commits = pd.DataFrame(rows)
commits['commit_date'] = pd.to_datetime(commits['commit_date'], utc=True, errors='coerce')
commits['month']       = commits['commit_date'].dt.strftime('%Y-%m')

commit_monthly = (commits.groupby(['repo','month'], as_index=False)
                  .agg(monthly_commits=('sha','count')))

monthly = pd.read_csv(PROCESSED / 'repository_month_metrics.csv')
monthly = monthly.merge(commit_monthly, on=['repo','month'], how='left')
monthly['monthly_commits'] = monthly['monthly_commits'].fillna(0)

save_csv_checkpoint(monthly,
                    PROCESSED / 'repository_month_metrics_with_controls.csv',
                    '11_with_controls')
checkpoint('11_complete', {'commit_rows': len(commits), 'repos_done': len(done)})
display(monthly[['repo','cohort','month','months_before_t0',
                 'requirement_issues','monthly_commits']].head())
