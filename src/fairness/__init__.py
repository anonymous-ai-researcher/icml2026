"""
Fairness metrics and verification.
"""

from .envy import (
    compute_bundle_value,
    compute_envy_matrix,
    is_envy_free,
    compute_ef1_violations,
    is_ef1,
    compute_efx_violations,
    is_efx,
    compute_proportionality_violations,
    compute_nash_welfare,
    compute_utilitarian_welfare,
    compute_egalitarian_welfare,
    FairnessMetrics,
)

__all__ = [
    "compute_bundle_value",
    "compute_envy_matrix",
    "is_envy_free",
    "compute_ef1_violations",
    "is_ef1",
    "compute_efx_violations",
    "is_efx",
    "compute_proportionality_violations",
    "compute_nash_welfare",
    "compute_utilitarian_welfare",
    "compute_egalitarian_welfare",
    "FairnessMetrics",
]
