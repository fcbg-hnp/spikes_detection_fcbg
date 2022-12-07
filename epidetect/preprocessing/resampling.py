from collections import Counter

"""
Module that defines different resampling strategies
"""


def smote_over_sampling_rate(y, non_spikes_label):
    c = Counter(y)
    return {1: int(c[1] * 1.3), non_spikes_label: c[non_spikes_label]}


def random_over_sampling_rate(y, orate, non_spikes_label):
    c = Counter(y)
    return {1: int(c[1] * orate), non_spikes_label: c[non_spikes_label]}


def random_under_sampling_rate(y, urate, non_spikes_label):
    c = Counter(y)
    return {1: c[1], non_spikes_label: int(c[non_spikes_label] * urate)}