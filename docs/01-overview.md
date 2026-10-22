# 01 — Overview

FinPulse demonstrates a customer retention decision workflow using a public Telco churn dataset as a proxy. It is a research and demonstration system; its effects, costs, and financial returns are scenario assumptions, not measured results.

The supported full paths are (1) training and evaluation on labeled history, with a held-out cohort, and (2) inference on a new unlabeled cohort using a stored model bundle. The dashboard reads the held-out evaluation snapshot rather than training when a page opens. It requires a configured shared access token.

The workflow validates data, fits preprocessing and segmentation on training rows, estimates churn and CLV, allocates scenario actions under a budget, and reports gross and risk-adjusted scenario economics. KMeans labels are neutral `Segment_i` because cluster IDs do not have stable business semantics.

Before use with real customers, define a prediction horizon and observation cutoff, validate the model on later data, obtain evidence for treatment effects and costs, and put the dashboard behind individual authentication and authorization.
