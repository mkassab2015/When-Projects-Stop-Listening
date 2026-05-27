# 08_evaluate_and_merge_manual_labels.py
# Purpose:
#   Evaluate automated classifier against manual labels and merge.
#   Manual labels are authoritative for sampled issues; auto labels are
#   used as fallback for unsampled issues.
#
# No logic changes from previous version.

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score

issues         = pd.read_csv(PROCESSED / 'issues_auto_classified.csv')
completed_path = MANUAL / 'manual_coding_completed.csv'
recode_path    = MANUAL / 'manual_recode_sample.csv'

if not completed_path.exists():
    print('[WARNING] manual_coding_completed.csv not found — using auto labels only.')
    print('  Document in Threats to Validity: classifier precision/recall unknown.')
    issues['final_is_requirement'] = issues['auto_is_requirement']
    issues['final_req_type']       = issues['auto_req_type']
    for c in ['manual_is_requirement','manual_req_type',
              'manual_has_expected_behavior','manual_has_rationale',
              'manual_has_acceptance','manual_is_actionable']:
        issues[c] = np.nan
    with open(RESULTS / 'classifier_validation_notes.txt', 'w') as f:
        f.write('Manual validation not performed. Auto labels used throughout.\n')
    save_csv_checkpoint(issues, PROCESSED / 'issues_final_classified.csv',
                        '08_auto_only')
else:
    manual = pd.read_csv(completed_path)
    manual['manual_is_requirement'] = pd.to_numeric(
        manual['manual_is_requirement'], errors='coerce')
    valid = manual.dropna(subset=['manual_is_requirement']).copy()
    valid['manual_is_requirement'] = valid['manual_is_requirement'].astype(int)

    print(f'Manual coded: {len(valid)} | Req prevalence: {valid["manual_is_requirement"].mean():.1%}')

    y_true, y_pred = valid['manual_is_requirement'], valid['auto_is_requirement'].astype(int)
    print(classification_report(y_true, y_pred, digits=3))
    print(confusion_matrix(y_true, y_pred))

    report_dict = classification_report(y_true, y_pred, output_dict=True)
    save_csv_checkpoint(pd.DataFrame(report_dict).T,
                        RESULTS / 'classifier_validation_overall.csv',
                        '08_classifier_validation_overall')

    # Per-stratum validation
    strata_rows = []
    for (cohort, period), grp in valid.groupby(['cohort','period_half'], observed=True):
        if len(grp) < 10: continue
        r = classification_report(grp['manual_is_requirement'],
                                   grp['auto_is_requirement'].astype(int),
                                   output_dict=True, zero_division=0)
        strata_rows.append({'cohort': cohort, 'period': period, 'n': len(grp),
                            'precision': round(r['1']['precision'], 3),
                            'recall':    round(r['1']['recall'],    3),
                            'f1':        round(r['1']['f1-score'],  3)})
    save_csv_checkpoint(pd.DataFrame(strata_rows),
                        RESULTS / 'classifier_validation_by_stratum.csv')

    # Intra-rater kappa
    if recode_path.exists():
        recode = pd.read_csv(recode_path)
        recode['manual_is_requirement_recode'] = pd.to_numeric(
            recode['manual_is_requirement_recode'], errors='coerce')
        rv = recode.dropna(subset=['manual_is_requirement_recode']).merge(
            valid[['issue_id','manual_is_requirement']], on='issue_id', how='inner')
        if len(rv) >= 10:
            kappa = cohen_kappa_score(rv['manual_is_requirement'],
                                      rv['manual_is_requirement_recode'])
            print(f'Intra-rater kappa: {kappa:.3f}')
            with open(RESULTS / 'intra_rater_reliability.txt', 'w') as f:
                f.write(f'kappa = {kappa:.3f}\nn_recoded = {len(rv)}\n')

    keep_manual = ['issue_id','manual_is_requirement','manual_req_type',
                   'manual_has_expected_behavior','manual_has_rationale',
                   'manual_has_acceptance','manual_is_actionable']
    issues = issues.merge(manual[keep_manual], on='issue_id', how='left')
    issues['final_is_requirement'] = issues['manual_is_requirement'].where(
        issues['manual_is_requirement'].notna(), issues['auto_is_requirement']).astype(int)
    issues['final_req_type'] = issues['manual_req_type'].where(
        issues['manual_req_type'].notna() & (issues['manual_req_type'] != ''),
        issues['auto_req_type'])

    save_csv_checkpoint(issues, PROCESSED / 'issues_final_classified.csv',
                        '08_issues_final_classified')
    print(issues['final_is_requirement'].value_counts())
