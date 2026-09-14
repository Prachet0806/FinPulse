# simulation/roi_calculator.py

import logging
import math

logger = logging.getLogger(__name__)


def compute_roi(impact):
    """
    Calculate ROI from impact metrics.

    - cost == 0 + gain > 0 -> inf
    - cost == 0 + gain == 0 -> 0.0
    - None/NaN/negative guards -> 0.0 with warning
    """
    if not isinstance(impact, dict):
        logger.warning("ROI quarantine: impact is not a dict")
        return 0.0
    cost = impact.get("total_strategy_cost")
    gain = impact.get("total_retention_gain", impact.get("net_benefit", 0.0))
    try:
        cost_f = float(cost)
        gain_f = float(gain)
    except (TypeError, ValueError):
        logger.warning("ROI quarantine: non-numeric cost/gain")
        return 0.0
    if math.isnan(cost_f) or math.isnan(gain_f) or cost_f < 0:
        logger.warning("ROI quarantine: invalid cost=%s gain=%s", cost, gain)
        return 0.0

    if cost_f == 0:
        if gain_f > 0:
            return float("inf")
        return 0.0

    roi = impact.get("net_benefit", gain_f - cost_f) / cost_f
    if not math.isfinite(roi):
        return 0.0
    return float(roi)
