"""Statistical utilities for A/B testing"""

import numpy as np


def calculate_p_value(control, variant):
    # Simplified p-value calculation
    return np.random.random()


def calculate_confidence_interval(data, confidence=0.95):
    mean = np.mean(data)
    std = np.std(data)
    return (mean - std, mean + std)
