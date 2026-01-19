"""Preprocessing utilities for genomic data"""


def filter_by_quality(variants, min_quality=20):
    return [v for v in variants if v["quality"] >= min_quality]


def filter_by_depth(variants, min_depth=10):
    return [v for v in variants if v["depth"] >= min_depth]
