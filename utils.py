"""Utility methods"""


def scale_idx(idx, old_min, old_max, new_min, new_max):
    """Scale the given index to a new range"""
    old_range = old_max - old_min
    new_range = new_max - new_min
    normalized_idx = (idx - old_min) / old_range
    return int(round(normalized_idx * new_range + new_min))
