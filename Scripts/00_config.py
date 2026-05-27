# 00_config.py
# Purpose:
#   Configure the study, mount Google Drive, install packages,
#   and define all shared constants, helpers, and the domain taxonomy.
#
# Fixes in this version:
#   - infer_domain() now uses regex word-boundary matching (\b) instead of
#     plain substring matching to prevent false positives such as 'ml' matching
#     'email', 'ai' matching 'trail', 'cli' matching 'client', etc.
#   - CODEBOOK embedded as a DataFrame literal so script 07 works in Colab
#     without depending on a local file path.

!pip -q install tqdm scikit-learn statsmodels lifelines python-dateutil scipy seaborn

import os
import re
import time
import json
import warnings
from pathlib import Path
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── Google Drive ──────────────────────────────────────────────────────────────
try:
    from google.colab import drive
    drive.mount('/content/drive')
    ROOT = Path('/content/drive/MyDrive/requirements_decay_study')
except Exception:
    ROOT = Path('./requirements_decay_study')

RAW         = ROOT / 'raw'
PROCESSED   = ROOT / 'processed'
MANUAL      = ROOT / 'manual_coding'
RESULTS     = ROOT / 'results'
FIGURES     = ROOT / 'figures'
CHECKPOINTS = ROOT / 'checkpoints'

for p in [RAW, PROCESSED, MANUAL, RESULTS, FIGURES, CHECKPOINTS]:
    p.mkdir(parents=True, exist_ok=True)

# ── Study parameters ──────────────────────────────────────────────────────────
# Start pilot with 10+10; change to 100+100 after full pipeline succeeds.
N_ABANDONED = 100
N_ACTIVE    = 100
OBS_MONTHS  = 24   # months of observation window before T0

# RUN_DATE is fixed for reproducibility. All cutoffs derive from it.
RUN_DATE = datetime(2026, 5, 10, tzinfo=timezone.utc)

# Abandoned: no push for ≥12 months before RUN_DATE.
ABANDONED_CUTOFF = RUN_DATE - relativedelta(months=12)

# Active: pushed within 3 months of RUN_DATE.
ACTIVE_RECENT_CUTOFF = RUN_DATE - relativedelta(months=3)

# Minimum human commits in the 24-month window to qualify as "previously active".
MIN_COMMITS_BEFORE_T0 = 20

# Nearest-neighbour distance threshold for poor-match flagging (robustness check 4).
MATCH_DISTANCE_THRESHOLD = 2.0

# Manual validation sample sizes.
MANUAL_VALIDATION_SIZE = 200 if N_ABANDONED <= 10 else 1000

# Languages searched.
LANGUAGES = [
    'Python', 'JavaScript', 'TypeScript', 'Java',
    'Go', 'Ruby', 'PHP', 'C++', 'C#', 'Rust',
]

# ── Domain taxonomy ───────────────────────────────────────────────────────────
# FIX: Every keyword is matched as a whole word using regex \b boundaries.
# This prevents false positives: 'ml' no longer matches 'email',
# 'ai' no longer matches 'trail', 'cli' no longer matches 'client',
# 'db' no longer matches 'debug', 'api' no longer matches 'capabilities'.
#
# Domains are checked in priority order; first match wins.
# 'other' is the catch-all and must remain last.

DOMAIN_PATTERNS = {
    # All patterns use \b word-boundary anchors only.
    # FIX: removed malformed \b(^|[^a-z])X([^a-z]|$)\b constructs that
    # combined \b with character-class lookarounds, producing patterns that
    # never matched. \b alone is sufficient and correct: \bml\b matches
    # standalone "ml" but not "email" or "html".
    'web':         [r'\bweb\b', r'\bhttp\b', r'\bapi\b', r'\brest\b',
                    r'\bgraphql\b', r'\bfrontend\b', r'\bbackend\b',
                    r'\bserver\b', r'\bdjango\b', r'\bflask\b',
                    r'\bexpress\b', r'\brails\b', r'\blaravel\b',
                    r'\bspring\b', r'\bfastapi\b'],
    'data_ml':     [r'\bmachine.learning\b', r'\bdeep.learning\b',
                    r'\bml\b',           # FIX: was \b(^|[^a-z])ml([^a-z]|$)\b
                    r'\bai\b',           # FIX: was \b(^|[^a-z])ai([^a-z]|$)\b
                    r'\bdata.science\b', r'\bnlp\b', r'\bpytorch\b',
                    r'\btensorflow\b', r'\bscikit\b', r'\bpandas\b',
                    r'\bspark\b', r'\banalytics\b'],
    'devops':      [r'\bdevops\b',
                    r'\bci\b',           # FIX: was \b(^|[^a-z])ci([^a-z]|$)\b
                    r'\bcd\b',           # FIX: was \b(^|[^a-z])cd([^a-z]|$)\b
                    r'\bdocker\b', r'\bkubernetes\b', r'\bk8s\b',
                    r'\binfrastructure\b', r'\bterraform\b',
                    r'\bansible\b', r'\bhelm\b', r'\bmonitoring\b',
                    r'\bobservability\b'],
    'cli_tool':    [r'\bcli\b',          # FIX: was \b(^|[^a-z])cli([^a-z]|$)\b
                    r'\bcommand.line\b', r'\bterminal\b', r'\bshell\b',
                    r'\butility\b', r'\bautomation\b'],
    'library_sdk': [r'\blibrary\b',
                    r'\bsdk\b',          # FIX: was \b(^|[^a-z])sdk([^a-z]|$)\b
                    r'\bframework\b', r'\bpackage\b', r'\bplugin\b',
                    r'\bextension\b', r'\bmiddleware\b'],
    'mobile':      [r'\bandroid\b',
                    r'\bios\b', r'\bswiftui\b',  # swiftui added for precision
                    r'\bmobile\b', r'\breact.native\b', r'\bflutter\b',
                    r'\bswift\b', r'\bkotlin\b'],
    'database':    [r'\bdatabase\b',
                    r'\bdb\b',           # FIX: was \b(^|[^a-z])db([^a-z]|$)\b
                    r'\bsql\b',          # FIX: was \b(^|[^a-z])sql([^a-z]|$)\b
                    r'\bnosql\b', r'\borm\b', r'\bcache\b', r'\bredis\b',
                    r'\bpostgres\b', r'\bmysql\b', r'\bmongodb\b'],
    'security':    [r'\bsecurity\b', r'\bcryptography\b', r'\bauth\b',
                    r'\boauth\b', r'\bvulnerability\b', r'\bpentest\b',
                    r'\bencryption\b'],
    'other':       [],   # catch-all — must be last
}


def infer_domain(topics: list, description: str) -> str:
    """
    Assign a coarse domain label using whole-word regex matching on
    GitHub topics and repository description.

    Uses word-boundary patterns to prevent substring false positives.
    Returns the first matching domain or 'other'.
    """
    text = ' '.join(topics or []) + ' ' + (description or '')
    text = text.lower()
    for domain, patterns in DOMAIN_PATTERNS.items():
        if domain == 'other':
            continue
        if any(re.search(p, text) for p in patterns):
            return domain
    return 'other'


# ── Codebook (embedded — no external file dependency) ────────────────────────
# Embedding the codebook here means script 07 works in any Colab environment
# without relying on a local path that may not exist.

CODEBOOK = pd.DataFrame([
    {
        'column':        'manual_is_requirement',
        'allowed_values': '0/1',
        'definition':    ('1 if the issue expresses a desired behavior, user need, '
                          'quality expectation, compatibility constraint, or '
                          'bug-as-unmet-expectation.'),
    },
    {
        'column':        'manual_req_type',
        'allowed_values': 'feature|quality|compatibility|bug_expectation|documentation|non_requirement|unclear',
        'definition':    'Primary category.',
    },
    {
        'column':        'manual_has_expected_behavior',
        'allowed_values': '0/1',
        'definition':    '1 if the issue states what the system should do.',
    },
    {
        'column':        'manual_has_rationale',
        'allowed_values': '0/1',
        'definition':    '1 if the issue explains why the change matters.',
    },
    {
        'column':        'manual_has_acceptance',
        'allowed_values': '0/1',
        'definition':    ('1 if the issue includes examples, steps, screenshots, '
                          'or testable success conditions.'),
    },
    {
        'column':        'manual_is_actionable',
        'allowed_values': '0/1',
        'definition':    '1 if a maintainer could act without major clarification.',
    },
    {
        'column':        'manual_notes',
        'allowed_values': 'text',
        'definition':    'Optional notes.',
    },
])

# ── GitHub API credentials ────────────────────────────────────────────────────
GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN', '')
HEADERS = {'Accept': 'application/vnd.github+json'}
if GITHUB_TOKEN:
    HEADERS['Authorization'] = f'Bearer {GITHUB_TOKEN}'

# ── Checkpoint utilities ──────────────────────────────────────────────────────

def checkpoint(name: str, info: dict = None):
    """Write a timestamped JSON checkpoint to Google Drive."""
    info = info or {}
    payload = {
        'checkpoint':    name,
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'root':          str(ROOT),
        **info,
    }
    out = CHECKPOINTS / f'{name}.json'
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, default=str)
    print(f'[CHECKPOINT] {out}')


def save_csv_checkpoint(df: pd.DataFrame, path, checkpoint_name: str = None,
                        index: bool = False):
    """Save a DataFrame to CSV and optionally write a checkpoint JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    print(f'[CSV SAVED] {path} | rows={len(df)}')
    if checkpoint_name:
        checkpoint(checkpoint_name, {'output_csv': str(path), 'rows': int(len(df))})


def warn_search_cap(items: list, query: str):
    """Log a warning when a GitHub search query returns ≥1000 results."""
    if len(items) >= 1000:
        msg = (f'[WARNING] Search hit 1000-result cap.\n'
               f'  Query: {query}\n'
               f'  Results may be incomplete.')
        print(msg)
        cap_log = CHECKPOINTS / 'search_cap_warnings.txt'
        with open(cap_log, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')


# ── Startup summary ───────────────────────────────────────────────────────────
print('Project root:            ', ROOT)
print('Sample size (each arm):  ', N_ABANDONED, '+', N_ACTIVE)
print('Observation window:      ', OBS_MONTHS, 'months')
print('Abandonment cutoff:      ', ABANDONED_CUTOFF.date())
print('Active cutoff:           ', ACTIVE_RECENT_CUTOFF.date())
print('MIN_COMMITS_BEFORE_T0:   ', MIN_COMMITS_BEFORE_T0)
print('MATCH_DISTANCE_THRESHOLD:', MATCH_DISTANCE_THRESHOLD)
print('GitHub token loaded:     ', bool(GITHUB_TOKEN))

checkpoint('00_config', {
    'N_ABANDONED':              N_ABANDONED,
    'N_ACTIVE':                 N_ACTIVE,
    'OBS_MONTHS':               OBS_MONTHS,
    'RUN_DATE':                 RUN_DATE.isoformat(),
    'ABANDONED_CUTOFF':         ABANDONED_CUTOFF.isoformat(),
    'ACTIVE_RECENT_CUTOFF':     ACTIVE_RECENT_CUTOFF.isoformat(),
    'MATCH_DISTANCE_THRESHOLD': MATCH_DISTANCE_THRESHOLD,
    'MANUAL_VALIDATION_SIZE':   MANUAL_VALIDATION_SIZE,
    'github_token_loaded':      bool(GITHUB_TOKEN),
})
