# decision_engine/strategy_library.py

# Source: static-estimate, needs A/B validation.
# TODO: replace with per-segment effectiveness learned from experiments.
STRATEGIES = {
    "cashback_offer": {
        "cost": 200,
        "effectiveness": 0.25,
        "risk_penalty": 0.0,
        "cap_percent": 0.20,
    },
    "fee_waiver": {
        "cost": 100,
        "effectiveness": 0.20,
        "risk_penalty": 20.0,
        "cap_percent": 0.30,
    },
    "credit_limit_increase": {
        "cost": 0,
        "effectiveness": 0.18,
        "risk_penalty": 50.0,  # Highly risky if they churn
        "cap_percent": 0.15,
    },
    "loan_offer": {
        "cost": 50,
        "effectiveness": 0.22,
        "risk_penalty": 10.0,
        "cap_percent": 0.10,
    },
    "no_action": {
        "cost": 0,
        "effectiveness": 0.0,
        "risk_penalty": 0.0,
        "cap_percent": 1.0,  # Uncapped fallback
    },
}
