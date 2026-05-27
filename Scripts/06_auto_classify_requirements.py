# 06_auto_classify_requirements.py
# Purpose:
#   Classify issues as requirement-like using transparent heuristic rules
#   and compute per-issue quality indicators.
#
# No logic changes from previous version.
# Retained fixes: vagueness_density, actionability components, bug_expectation
# as separate type, OR-classifier design documented.

import re
import numpy as np
import pandas as pd

issues = pd.read_csv(RAW / 'issues_raw.csv')

# ── Pattern lists ──────────────────────────────────────────────────────────────

REQ_LABEL_PATTERNS = [
    r'\benhancement\b', r'\bfeature\b', r'\bproposal\b', r'\brequest\b',
    r'\busability\b', r'\bux\b', r'\bperformance\b', r'\bsecurity\b',
    r'\bcompatibility\b', r'\broadmap\b', r'\bwish\b', r'\bimprovement\b',
]

REQ_TEXT_PATTERNS = [
    r'\bshould support\b', r'\bplease add\b', r'\bfeature request\b',
    r'\bit would be (nice|useful|helpful|great)\b',
    r'\busers? (need|want|expect)\b', r'\bexpected behavior\b',
    r'\bactual behavior\b', r'\bshould be able to\b',
    r'\ballow users? to\b', r'\bsupport for\b', r'\bwould (like|love) to\b',
    r'\bmissing (feature|support|option)\b',
    r'\bcurrently (impossible|not possible|unsupported)\b',
    r'\bas a user\b', r'\buse case\b',
]

VAGUE_PATTERNS = [
    r'\bmaybe\b', r'\bperhaps\b', r'\bsomehow\b', r'\bnot sure\b',
    r'\bideally\b', r'\bprobably\b', r'\bkind of\b', r'\bsort of\b',
]

# Rationale indicators (Aranda & Venolia 2009)
RATIONALE_PATTERNS = [
    r'\bbecause\b', r'\bso that\b', r'\bin order to\b', r'\buse case\b',
    r'\bmotivation\b', r'\bthe reason\b',
    r'\bthis (would|will) (help|allow|enable)\b',
]

# Acceptance/expected-behaviour indicators (Zimmermann et al. 2010)
ACCEPTANCE_PATTERNS = [
    r'\bgiven\b.{1,60}\bwhen\b.{1,60}\bthen\b',
    r'\bexpected (behavior|behaviour|result|output)\b',
    r'\bsteps to reproduce\b', r'\bexample\b',
    r'\bscreenshot\b', r'\bscreencast\b',
    r'\bminimal (reproducible|working) example\b',
]

TYPE_PRIORITY = [
    ('security',        [r'security', r'vulnerability', r'cve', r'auth',
                         r'permission', r'privacy', r'exploit']),
    ('performance',     [r'performance', r'slow', r'latency', r'throughput',
                         r'memory', r'cpu', r'speed', r'bottleneck']),
    ('usability',       [r'usability', r'\bux\b', r'confusing', r'user.friend',
                         r'accessibility', r'a11y', r'hard to use']),
    ('compatibility',   [r'compatibility', r'backward.compat', r'\bplatform\b',
                         r'dependency', r'\bversion\b', r'interop']),
    ('documentation',   [r'documentation', r'\bdocs\b', r'\breadme\b',
                         r'tutorial', r'example missing']),
    ('bug_expectation', [r'expected behavior', r'expected behaviour',
                         r'actual behavior', r'actual behaviour',
                         r'should (work|behave)', r'regression']),
    ('feature',         [r'enhancement', r'feature', r'please add',
                         r'support for', r'allow', r'new (option|parameter)']),
]


def clean_text(s):
    if pd.isna(s): return ''
    s = str(s)
    s = re.sub(r'```.*?```', ' ', s, flags=re.DOTALL)
    s = re.sub(r'`[^`]+`', ' ', s)
    s = re.sub(r'https?://\S+', ' ', s)
    s = re.sub(r'<!--.*?-->', ' ', s, flags=re.DOTALL)
    return re.sub(r'\s+', ' ', s).strip()

def contains_any(text, patterns):
    t = str(text).lower()
    return any(re.search(p, t, re.I) for p in patterns)

def count_matches(text, patterns):
    t = str(text).lower()
    return sum(1 for p in patterns if re.search(p, t, re.I))

def classify_type(text, labels):
    combined = (str(labels) + ' ' + str(text)).lower()
    for typ, pats in TYPE_PRIORITY:
        if contains_any(combined, pats):
            return typ
    return 'unclear'


for c in ['created_at', 'closed_at', 't0']:
    issues[c] = pd.to_datetime(issues[c], utc=True, errors='coerce')

issues['text']           = (issues['title'].fillna('') + ' ' +
                             issues['body'].fillna('')).apply(clean_text)
issues['text_len_words'] = issues['text'].apply(lambda x: len(x.split()))

issues['label_req_signal']    = issues['labels'].fillna('').apply(
    lambda x: contains_any(x, REQ_LABEL_PATTERNS))
issues['text_req_signal']     = issues['text'].apply(
    lambda x: contains_any(x, REQ_TEXT_PATTERNS))
issues['auto_is_requirement'] = (issues['label_req_signal'] |
                                  issues['text_req_signal']).astype(int)
issues['auto_req_type']       = issues.apply(
    lambda r: classify_type(r['text'], r['labels']), axis=1)

issues['has_rationale']       = issues['text'].apply(
    lambda x: contains_any(x, RATIONALE_PATTERNS)).astype(int)
issues['has_acceptance']      = issues['text'].apply(
    lambda x: contains_any(x, ACCEPTANCE_PATTERNS)).astype(int)
issues['has_expected_beh']    = issues['text'].apply(
    lambda x: contains_any(x, [r'expected (behavior|behaviour|result)'])).astype(int)

issues['vagueness_count']     = issues['text'].apply(
    lambda x: count_matches(x, VAGUE_PATTERNS))
issues['vagueness_density']   = (
    issues['vagueness_count'] /
    issues['text_len_words'].replace(0, np.nan)
).fillna(0)

issues['actionability_length'] = (issues['text_len_words'] >= 30).astype(int)
issues['actionability_score']  = (
    issues['actionability_length'] + issues['has_rationale'] +
    issues['has_acceptance'] + issues['label_req_signal'].astype(int)
)

save_csv_checkpoint(issues, PROCESSED / 'issues_auto_classified.csv',
                    '06_issues_auto_classified')

with open(RESULTS / 'classifier_design_notes.txt', 'w') as f:
    f.write(
        'Classifier: OR(label_signal, text_signal) — high recall by design.\n'
        'Precision unknown until manual validation (script 08).\n'
        'vagueness_density = vagueness_count / word_count.\n'
        'actionability_score: equal-weight composite (0–4), exploratory.\n'
        'bug_expectation kept as a separate type.\n'
    )

print('Classification summary:')
print(issues['auto_is_requirement'].value_counts())
print(issues['auto_req_type'].value_counts())
