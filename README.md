# 💳 FinPulse — Customer Decision Intelligence Platform

End-to-end analytics system that transforms customer data into retention strategies with quantified financial impact for fintech and banking organizations.

---

## 🎯 Executive Summary

Financial institutions struggle to allocate retention budgets efficiently, often applying blanket strategies that waste resources or fail to target high-value clients at critical risk. 

**FinPulse** provides data-driven decision support to solve this optimization problem. It is designed for analytics product teams, customer success leaders, and financial executives. By bridging the gap between predictive modeling and business logic, FinPulse identifies high-value customers at risk and automatically recommends optimal, ROI-positive retention actions.

---

## 💡 Key Capabilities

### 🔍 Customer Intelligence
* **Behavioral Segmentation**: Unsupervised clustering (K-Means with automated K-selection) to identify distinct customer personas based on value and engagement.
* **Churn Prediction**: Advanced classification modeling to predict the probability of customer attrition.
* **Financial CLV Estimation**: Retention-based Customer Lifetime Value calculation (`CLV = (ARPU * Retention) / (1 + Discount - Retention)`).

### 🎯 Decision Optimization (V2 Enterprise Engine)
* **4-Layer Decision Framework**: Sequential processing across Base Valuation, Risk Adjustment, Eligibility Rules, and Portfolio Allocation.
* **Risk-Adjusted ROI**: Strategies are penalized based on the baseline churn risk of the customer, ensuring conservative allocation for high-risk targets.
* **Portfolio Allocation**: Global optimization subject to a monthly budget and individual strategy capacity limits (caps).

### 📈 Business Impact
* **Revenue Impact Simulation**: Quantifies the total retention gain and strategy costs across the portfolio.
* **Interactive Dashboard**: An industry-grade executive dashboard for visualizing risk matrixes, strategy allocations, and overall ROI.

---

## 🏗️ Architecture Overview

```text
[Raw Data] --> (Ingestion & Validation)
                       ↓
[Feature Engineering] --> (Domain Transformation, Categorical Encoding)
                       ↓
[Customer Intelligence Layer]
  ├── Segmentation Engine (Clustering)
  ├── Predictive Engine (Churn Model)
  └── Valuation Engine (CLV Calculation)
                       ↓
[Decision Engine] 
  ├── Risk Adjustment (Probability-scaled Penalties)
  ├── Eligibility Masking (Tenure & Risk-based Rules)
  └── Portfolio Allocator (ROI Density vs. Global Budget)
                       ↓
[Impact Simulation] --> (ROI Calculation & Scenario Analysis)
                       ↓
[Executive Dashboard] --> (Interactive UI & Reporting)
```

---

## 🔄 End-to-End Workflow

The FinPulse system operates on a streamlined, modular pipeline:

1. **Data Ingestion**: Loads raw legacy data and validates schema requirements.
2. **Domain Transformation**: Maps base features to financial domain concepts (e.g., Monthly Spend, Account Tenure).
3. **Segmentation**: Dynamically scales features and clusters the portfolio into actionable segments.
4. **Prediction**: Trains and scores the portfolio using a RandomForest classifier to establish baseline risk.
5. **Decision Optimization**: Multi-layer engine evaluates strategies against risk/value profiles, masks ineligible targets, and allocates assignments based on ROI density and global budget constraints.
6. **Impact Simulation**: Calculates the systemic financial impact and projected ROI of the recommended actions.
7. **Reporting**: Serves the insights via a high-performance Streamlit/Plotly web framework.

---

## 📊 Dataset & Domain Adaptation

FinPulse demonstrates the ability to adapt raw data sets to specific industry domains.

* **Dataset Source**: Initially built utilizing a standard Telco Customer Dataset (serving as a proxy for subscription-based service data).
* **Fintech Mapping**: The pipeline programmatically translates telco-specific fields into universal financial metrics (e.g., `MonthlyCharges` -> `monthly_spend`, `tenure` -> `account_tenure_months`).
* **Economic Scaling (10x Multiplier)**: Because Telco ARPU (~$70) does not match high-value Fintech economics, a **10x multiplier** is applied to `monthly_spend` and `lifetime_value` during Feature Engineering. This scales Customer Lifetime Values (CLV) to the $2,000–$4,000 range, enabling the Decision Engine to realistically evaluate strictly positive ROI for $100–$200 premium retention strategies.
* **Assumptions**: The system assumes contractual or semi-contractual customer relationships where "churn" is a discrete event, mimicking SaaS, banking, or insurance models.

---

## 🛠️ Technology Stack

### 🧠 Analytics & Modeling
* **Pandas & NumPy**: High-performance vectorized data manipulation.
* **Scikit-Learn**: Machine learning (RandomForest, KMeans, StandardScaler) and validation metrics (Silhouette Score).
* **Joblib**: Model persistence and artifact registry.

### 📊 Visualization & UI
* **Streamlit**: Rapid web application framework for the executive interface.
* **Plotly Express**: Interactive, publication-quality data visualizations.

### ⚙️ Infrastructure
* **Python 3.9+**: Core execution environment.
* **Pytest**: Unit testing framework for ensuring logic stability.

---

## 🚀 How to Run the Project

### 1. Installation
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/Prachet0806/FinPulse.git
cd FinPulse
pip install -r requirements.txt
```

### 2. Pipeline Execution
Run the end-to-end analytical pipeline to generate models and output metrics. *Ensure your `PYTHONPATH` is set to the project root.*
```bash
# On Windows (PowerShell)
$env:PYTHONPATH="."
python scripts/run_pipeline.py

# On Unix/macOS
export PYTHONPATH="."
python scripts/run_pipeline.py
```

### 3. Dashboard Launch
Start the interactive executive dashboard:
```bash
streamlit run dashboard/app.py
```

---

## 📸 Sample Outputs / Screenshots

*(Include screenshots of your dashboard here by adding images to a `docs/images/` directory)*

### 📊 Executive Dashboard
`![Executive Dashboard Overview](docs/images/dashboard_main.png)`
*A comprehensive view of portfolio ROI and retention impact.*

### 📈 Risk vs. Value Matrix
`![Risk vs Value Scatter](docs/images/risk_value_matrix.png)`
*Plotly-powered scatter matrix identifying high-value customers at immediate risk.*


---

## 🏢 Business Use Cases

While FinPulse is configured for Fintech, the underlying architecture is highly extensible to other domains:

* **Banking & Credit Cards**: Optimizing credit limit increases vs. fee waivers to prevent account closures.
* **Insurance**: Identifying policyholders at risk of non-renewal and calculating the cost-benefit of premium discounts.
* **B2B SaaS / Subscription Businesses**: Allocating Customer Success Manager (CSM) outreach bandwidth to clients with the highest potential systemic impact.

---

## 🧭 Design Principles

* **ROI-Driven Optimization**: We don't just predict churn; we optimize the *preventative response* based strictly on financial utility.
* **Explainability**: Complex models are mapped back to business-friendly segment labels and clear strategic recommendations.
* **Scalability**: The processing and decision layers are fully vectorized, allowing for rapid execution across millions of rows without distributed computing overhead.

---

## ⚠️ Limitations & Assumptions

* **Static Effectiveness Rates**: Currently, the strategy `effectiveness` and `cost` metrics in the `strategy_library` are static estimates. In reality, these would require empirical backing via historical A/B testing.
* **First-Order Impact Only**: The simulation assumes mutually independent churn events and does not account for secondary network effects or broader macroeconomic shifts.
* **Cost of Capital**: The CLV calculation utilizes a simplified universal discount rate.

---

## 🛤️ Future Enhancements (V2 Roadmap)

* **Next-Best-Offer Engine**: Context-aware recommendations utilizing collaborative filtering alongside the expected-value engine.
* **Linear Programming Solver**: Migrating from the current greedy ROI-density allocator to a formal LP solver (e.g., PuLP/SciPy) for exact mathematical optimization under competing constraints.
* **MLOps Integration**: Migration from local `joblib` persistence to a centralized `MLflow` tracking server for comprehensive model registry and versioning.
