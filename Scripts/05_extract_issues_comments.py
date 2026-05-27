# 05_extract_issues_comments.py
# Purpose:
#   Extract GitHub issues and comments within each repository's observation window.
#
# Fix: paginate_search_chunked renamed to paginate_issue_search_chunked in 01.
# All other logic unchanged from previous version.

import time
import pandas as pd
from tqdm.auto import tqdm

repo_panel = pd.read_csv(PROCESSED / 'repo_panel.csv')

issues_path   = RAW / 'issues_raw.csv'
comments_path = RAW / 'comments_raw.csv'
done_path     = CHECKPOINTS / 'issues_extraction_done_repos.csv'

issue_rows, comment_rows = [], []
done = set()

if issues_path.exists():
    issue_rows = pd.read_csv(issues_path).to_dict('records')
if comments_path.exists():
    comment_rows = pd.read_csv(comments_path).to_dict('records')
if done_path.exists():
    done = set(pd.read_csv(done_path)['repo'].dropna().unique())


def get_issue_comments(comments_url):
    return paginate(comments_url, params={}, max_pages=5)


for _, rr in tqdm(repo_panel.iterrows(), total=len(repo_panel),
                  desc='Extracting issues/comments'):
    full = rr['full_name']
    if full in done:
        continue

    start = pd.to_datetime(rr['window_start'], utc=True).to_pydatetime()
    end   = pd.to_datetime(rr['window_end'],   utc=True).to_pydatetime()

    try:
        items = paginate_issue_search_chunked(     # renamed from paginate_search_chunked
            base_query  = f'repo:{full} type:issue',
            start_date  = start,
            end_date    = end,
            sort        = 'created',
            order       = 'asc',
            chunk_weeks = 8,
            max_pages   = 10,
        )

        if len(items) >= 1000:
            print(f'  [WARNING] {full}: {len(items)} issues — residual cap risk.')

        for it in [i for i in items if 'pull_request' not in i]:
            issue_rows.append({
                'repo':           full,
                'pair_id':        rr['pair_id'],
                'cohort':         rr['cohort'],
                'language':       rr['language'],
                'domain':         rr.get('domain', 'other'),
                't0':             rr['t0'],
                'issue_id':       it.get('id'),
                'number':         it.get('number'),
                'title':          it.get('title'),
                'body':           it.get('body'),
                'state':          it.get('state'),
                'created_at':     it.get('created_at'),
                'updated_at':     it.get('updated_at'),
                'closed_at':      it.get('closed_at'),
                'author':         (it.get('user') or {}).get('login'),
                'comments_count': it.get('comments', 0),
                'labels':         '|'.join(
                    lab.get('name', '') for lab in (it.get('labels') or [])
                ),
                'milestone':      (it.get('milestone') or {}).get('title'),
                'comments_url':   it.get('comments_url'),
                'html_url':       it.get('html_url'),
            })

            if it.get('comments', 0) > 0 and it.get('comments_url'):
                for c in get_issue_comments(it['comments_url']):
                    comment_rows.append({
                        'repo':               full,
                        'issue_id':           it.get('id'),
                        'issue_number':       it.get('number'),
                        'comment_id':         c.get('id'),
                        'comment_author':     (c.get('user') or {}).get('login'),
                        'comment_created_at': c.get('created_at'),
                        'comment_body':       c.get('body'),
                    })
                time.sleep(0.2)

        pd.DataFrame(issue_rows).to_csv(issues_path, index=False)
        pd.DataFrame(comment_rows).to_csv(comments_path, index=False)
        done.add(full)
        pd.DataFrame({'repo': sorted(done)}).to_csv(done_path, index=False)
        checkpoint('05_progress', {
            'last_repo': full, 'completed': len(done),
            'issues': len(issue_rows), 'comments': len(comment_rows),
        })
        time.sleep(1)

    except Exception as e:
        checkpoint('05_error', {'repo': full, 'error': str(e)})
        raise

issues_raw   = pd.DataFrame(issue_rows)
comments_raw = pd.DataFrame(comment_rows)
checkpoint('05_complete', {'issues': len(issues_raw), 'comments': len(comments_raw)})
print(f'Issues: {issues_raw.shape} | Comments: {comments_raw.shape}')
display(issues_raw.head())
