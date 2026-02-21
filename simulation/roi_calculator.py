# simulation/roi_calculator.py

def compute_roi(impact):
    """
    Calculate ROI from impact metrics.
    """

    cost = impact["total_strategy_cost"]

    if cost == 0:
        return 0

    roi = impact["net_benefit"] / cost
    return roi