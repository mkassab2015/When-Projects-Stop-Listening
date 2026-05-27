# 01_github_helpers.py
# Purpose:
#   Reusable GitHub REST API helpers: rate-limit handling, list pagination,
#   date-chunked search pagination (avoids the 1000-result cap), full repo
#   metadata fetching (including topics), and commit counting.
#
# No logic changes from previous version.
# Retained here as a clean standalone copy.

import time
import requests
from datetime import timedelta


def github_get(url, params=None, max_retries=5):
    """GET a GitHub API endpoint with rate-limit and server-error retries."""
    for attempt in range(max_retries):
        r = requests.get(url, headers=HEADERS, params=params)

        if r.status_code == 403 and 'rate limit' in r.text.lower():
            reset = int(r.headers.get('X-RateLimit-Reset', time.time() + 60))
            sleep_for = max(reset - int(time.time()) + 5, 30)
            print(f'  Rate limit. Sleeping {sleep_for}s …')
            time.sleep(sleep_for)
            continue

        if r.status_code in [500, 502, 503, 504]:
            time.sleep(2 ** attempt)
            continue

        if r.status_code >= 400:
            print(f'  GitHub error {r.status_code}: {url} | {r.text[:200]}')
            return None

        return r
    return None


def paginate(url, params=None, max_pages=10):
    """Paginate a list endpoint, returning all items across pages."""
    params = params or {}
    out = []
    for page in range(1, max_pages + 1):
        p = {**params, 'per_page': 100, 'page': page}
        r = github_get(url, params=p)
        if r is None:
            break
        data  = r.json()
        items = data.get('items', data) if isinstance(data, dict) else data
        if not items:
            break
        out.extend(items)
        if len(items) < 100:
            break
    return out


def paginate_search(url, params=None, max_pages=10):
    """
    Paginate a /search/ endpoint.

    Returns (items, hit_cap) where hit_cap=True means GitHub reported
    total_count ≥ 1000, indicating silent truncation.
    """
    params = params or {}
    out = []
    total_count = 0
    for page in range(1, max_pages + 1):
        p = {**params, 'per_page': 100, 'page': page}
        r = github_get(url, params=p)
        if r is None:
            break
        data = r.json()
        if page == 1:
            total_count = data.get('total_count', 0)
        items = data.get('items', [])
        if not items:
            break
        out.extend(items)
        if len(items) < 100:
            break
        time.sleep(0.5)
    return out, total_count >= 1000


def paginate_issue_search_chunked(base_query, start_date, end_date,
                                   sort='created', order='asc',
                                   chunk_weeks=8, max_pages=10):
    """
    Search GitHub issues in date sub-windows to avoid the 1000-result cap.

    Splits [start_date, end_date] into `chunk_weeks`-wide windows.
    If any window still reports total_count ≥ 1000, halves it recursively
    (minimum chunk: 1 week).

    Parameters
    ----------
    base_query  : str       GitHub search query without a date filter.
    start_date  : datetime
    end_date    : datetime
    chunk_weeks : int       Initial sub-window width in weeks.

    Returns
    -------
    list  Deduplicated raw issue dicts.
    """
    url       = 'https://api.github.com/search/issues'
    all_items = []
    seen_ids  = set()

    def _fetch(t_start, t_end, wsize):
        q = f'{base_query} created:{t_start.date()}..{t_end.date()}'
        items, hit_cap = paginate_search(
            url, params={'q': q, 'sort': sort, 'order': order},
            max_pages=max_pages,
        )
        if hit_cap and wsize > 1:
            mid = t_start + (t_end - t_start) / 2
            _fetch(t_start, mid, wsize // 2)
            _fetch(mid + timedelta(days=1), t_end, wsize // 2)
        else:
            if hit_cap:
                warn_search_cap(items, q)
            for it in items:
                iid = it.get('id')
                if iid and iid not in seen_ids:
                    seen_ids.add(iid)
                    all_items.append(it)

    delta  = timedelta(weeks=chunk_weeks)
    cursor = start_date
    while cursor < end_date:
        chunk_end = min(cursor + delta, end_date)
        _fetch(cursor, chunk_end, chunk_weeks)
        cursor = chunk_end + timedelta(days=1)
        time.sleep(1)

    return all_items


def get_repo_metadata(full_name):
    """
    Fetch full repository metadata including GitHub topics.
    Topics require the 'mercy-preview' Accept header.
    Returns dict or None.
    """
    url = f'https://api.github.com/repos/{full_name}'
    hdrs = {**HEADERS, 'Accept': 'application/vnd.github.mercy-preview+json'}
    r = requests.get(url, headers=hdrs)
    return r.json() if r.status_code == 200 else None


def get_recent_commit_count(full_name, since, until):
    """
    Count commits in [since, until] for a repository (capped at 1000).
    Used only for the binary pass/fail abandonment and activity checks,
    so exact counts above MIN_COMMITS_BEFORE_T0 are not needed.
    """
    url   = f'https://api.github.com/repos/{full_name}/commits'
    count = 0
    for page in range(1, 11):
        r = github_get(url, params={
            'since': since.isoformat(), 'until': until.isoformat(),
            'per_page': 100, 'page': page,
        })
        if r is None:
            break
        data = r.json()
        if not isinstance(data, list) or not data:
            break
        count += len(data)
        if len(data) < 100:
            break
    return count
