"""Numerical decisions frozen before confirmatory outcomes."""
import numpy as np


def absolute_adequacy(row):
    m=row['development']['clean']['metrics'];c=row['development']['clean']['collapse']
    return bool(m['mean_subject_macro_f1']>=.60 and m['per_stage_recall'][1]>=.20
                and min(m['per_stage_f1'])>=.10 and c['predicted_classes']==5 and c['largest_fraction']<.98)


def width_gain(large,small):
    a=large['development']['clean']['metrics'];b=small['development']['clean']['metrics']
    return bool(a['mean_subject_macro_f1']-b['mean_subject_macro_f1']>=.03
                and min(np.array(a['per_stage_f1'])-b['per_stage_f1'])>=-.10)


def exposure_gain(final,early):
    a=final['development']['clean']['metrics'];b=early['development']['clean']['metrics']
    return bool(a['mean_subject_macro_f1']-b['mean_subject_macro_f1']>=.03
                and a['per_stage_recall'][1]-b['per_stage_recall'][1]>=.10)


def routing_gain(routed,fixed):
    a=routed['development'];b=fixed['development']
    return bool(a['strong_local_HF']['metrics']['mean_subject_macro_f1']-b['strong_local_HF']['metrics']['mean_subject_macro_f1']>=.02
                and a['clean']['metrics']['mean_subject_macro_f1']-b['clean']['metrics']['mean_subject_macro_f1']>=-.02
                and min(np.array(a['clean']['metrics']['per_stage_f1'])-b['clean']['metrics']['per_stage_f1'])>=-.10)
