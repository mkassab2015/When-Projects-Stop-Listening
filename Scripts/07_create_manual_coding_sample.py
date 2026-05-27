# 07_create_manual_coding_sample.py
# Purpose:
#   Create a stratified sample of issues for manual classification.
#
# Fix in this version:
#   - Codebook is now read from the embedded CODEBOOK DataFrame defined in
#     00_config.py rather than from a local file path (/mnt/user-data/uploads/)
#     that does not exist in Google Colab. This makes the script fully portable.

import numpy as np
import pandas as pd

issues = pd.read_csv(PROCESSED / 'issues_auto_classified.csv')
issues['created_at'] = pd.to_datetime(issues['created_at'], utc=True, errors='coerce')
issues['t0']         = pd.to_datetime(issues['t0'],         utc=True, errors='coerce')

issues['months_before_t0'] = (
    (issues['t0'].dt.year  - issues['created_at'].dt.year)  * 12 +
    (issues['t0'].dt.month - issues['created_at'].dt.month)
)
issues['period_half'] = np.where(issues['months_before_t0'] > 12, 'early', 'late')

issues['type_bucket'] = issues['auto_req_type'].map({
    'feature': 'feature', 'quality': 'quality', 'security': 'quality',
    'performance': 'quality', 'usability': 'quality',
    'bug_expectation': 'bug_expectation', 'compatibility': 'quality',
    'documentation': 'other', 'unclear': 'other',
}).fillna('other')

# ── Stratified sampling ────────────────────────────────────────────────────────
STRATA_COLS = ['cohort', 'auto_is_requirement', 'period_half', 'type_bucket']
n_strata    = max(1, issues.groupby(STRATA_COLS).ngroups)
per_stratum = max(5, MANUAL_VALIDATION_SIZE // n_strata)

samples = []
for _, grp in issues.groupby(STRATA_COLS):
    samples.append(grp.sample(min(per_stratum, len(grp)), random_state=42))

sample = pd.concat(samples).drop_duplicates('issue_id')
if len(sample) > MANUAL_VALIDATION_SIZE:
    sample = sample.sample(MANUAL_VALIDATION_SIZE, random_state=42)

print(f'Manual coding sample: {len(sample)} issues')

MANUAL_COLS = [
    'manual_is_requirement', 'manual_req_type', 'manual_has_expected_behavior',
    'manual_has_rationale', 'manual_has_acceptance', 'manual_is_actionable',
    'manual_notes',
]
for c in MANUAL_COLS:
    sample[c] = ''

KEEP = ['repo', 'cohort', 'domain', 'issue_id', 'number', 'title', 'body',
        'labels', 'html_url', 'auto_is_requirement', 'auto_req_type',
        'text_len_words', 'period_half', 'type_bucket',
        'months_before_t0'] + MANUAL_COLS

save_csv_checkpoint(sample[KEEP], MANUAL / 'manual_coding_sample.csv',
                    '07_manual_coding_sample')

# ── Print codebook (from embedded config — no external file dependency) ────────
print('\nCODEBOOK (from 00_config.py):')
display(CODEBOOK)

print('\n' + '='*60)
print('STOP HERE.')
print('='*60)
print(f'Open:    {MANUAL}/manual_coding_sample.csv')
print('Fill the manual_* columns using the codebook above.')
print(f'Save as: {MANUAL}/manual_coding_completed.csv')
print()
print('OPTIONAL — intra-rater reliability:')
print('  After completing coding, recode 15% of issues after a 2-week gap.')
print(f'  Save as: {MANUAL}/manual_recode_sample.csv')
print('  Required columns: issue_id, manual_is_requirement_recode')
print('  Script 08 will compute Cohen\'s kappa automatically.')
print('='*60)
print('Then run: 08_evaluate_and_merge_manual_labels.py')
