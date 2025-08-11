🛡️ Insurance Renewal Prediction & Incentive Optimization
Predict which customers will renew their insurance policies and optimize agent incentives to maximize net revenue.

📌 Overview
This project delivers an end-to-end machine learning solution to help insurance companies:

Predict policy renewals using historical data.

Optimize agent incentives to maximize net revenue using a concave uplift model.

Deploy real-time predictions via a FastAPI + Docker microservice.

A 1% increase in renewal rates can mean millions in extra revenue. This system ensures data-driven decisions to achieve that.

🚀 Features
Data Cleaning & EDA: Missing value imputation, statistical analysis, and visualization.

Feature Engineering: Aggregated late payment counts, target encoding, and scalable preprocessing pipeline.

Model Training: Random Forest (bagging) & XGBoost (boosting) with hyperparameter tuning.

Metrics & Validation: AUC-ROC > 0.80, PR-AUC, calibration curve, and confusion matrix.

Incentive Optimization: Numerical optimization via SciPy to find the best incentive for each policy.

Deployment Ready: FastAPI REST API, Docker containerization, real-time predictions (<200ms).

Monitoring: Drift detection, retraining pipeline, and ROI tracking.

📊 Tech Stack
Languages & Libraries:

Python 3.10, Pandas, NumPy, Scikit-learn, XGBoost, Category Encoders, Matplotlib, Seaborn, SciPy

Deployment & Ops:

FastAPI, Uvicorn, Docker, Joblib, Pydantic

🧠 Machine Learning Pipeline
mermaid
Copy
Edit
flowchart TD
    A[Data Ingestion] --> B[Data Cleaning & Imputation]
    B --> C[Feature Engineering]
    C --> D[Model Training: RF + XGBoost]
    D --> E[Validation & Metrics]
    E --> F[Incentive Optimization]
    F --> G[FastAPI Deployment]
    G --> H[Monitoring & Drift Detection]
📈 Example Output
API Request:

json
Copy
Edit
{
  "age_in_days": 12000,
  "Income": 55000,
  "Count_3_6_months_late": 0,
  "Count_6_12_months_late": 1,
  "Count_more_than_12_months_late": 0,
  "application_underwriting_score": 0.85,
  "no_of_premiums_paid": 5,
  "sourcing_channel": "Online",
  "residence_area_type": "Urban",
  "premium": 41100,
  "perc_premium_paid_by_cash_credit": 0.63
}
API Response:

json
Copy
Edit
{
  "renewal_probability": 0.7557,
  "optimal_incentive": 795.8112
}
🛠️ Setup & Installation
Clone the repo

bash
Copy
Edit
git clone https://github.com/yourusername/insurance-renewal-prediction.git
cd insurance-renewal-prediction
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Run the API

bash
Copy
Edit
uvicorn app.main:app --reload
(Optional) Docker Deployment

bash
Copy
Edit
docker build -t insurance-renewal-api .
docker run -p 80:80 insurance-renewal-api
🏆 Results & Impact
Achieved AUC-ROC > 0.80 in renewal prediction.

Reduced wasted incentive spend by optimizing payouts per policy.

Demonstrated ₹X million net gain over a fixed incentive strategy.

Deployed production-ready API with monitoring and retraining capabilities.

📌 Future Improvements
Add Causal Uplift Modeling for better incentive allocation.

Integrate real-time streaming data for instant re-optimization.

Expand features with customer interaction data (calls, web visits).

# Insurance_Renewal_project
