from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

# Load the trained pipeline (preprocessor + model)
pipeline = joblib.load('models/final_pipeline.pkl')
preprocessor = pipeline['preprocessor']
model = pipeline['model']

# Initialize FastAPI app
app = FastAPI()

# Define request schema
class PolicyInput(BaseModel):
    age_in_days: float
    Income: float
    Count_3_6_months_late: int
    Count_6_12_months_late: int
    Count_more_than_12_months_late: int
    application_underwriting_score: float
    no_of_premiums_paid: int
    sourcing_channel: str
    residence_area_type: str
    premium: float
    perc_premium_paid_by_cash_credit: float

# Define net revenue function for incentive optimization
def net_revenue(I, p, premium):
    E = 10 * (1 - np.exp(-I / 400))
    dp = 0.20 * (1 - np.exp(-E / 5))
    return -((p + dp) * premium - I)

# API endpoint
@app.post("/predict-incentive")
def predict_incentive(policy: PolicyInput):
    # Convert input to dictionary
    data = policy.dict()

    # Create total late count
    data['total_late'] = (
        data['Count_3_6_months_late'] +
        data['Count_6_12_months_late'] +
        data['Count_more_than_12_months_late']
    )

    # Prepare input for model
    df = pd.DataFrame([{
    'age_in_days': data['age_in_days'],
    'Income': data['Income'],
    'application_underwriting_score': data['application_underwriting_score'],
    'no_of_premiums_paid': data['no_of_premiums_paid'],
    'sourcing_channel': data['sourcing_channel'],
    'residence_area_type': data['residence_area_type'],
    'premium': data['premium'],
    'total_late': data['total_late'],
    'perc_premium_paid_by_cash_credit': data['perc_premium_paid_by_cash_credit'] 
    }])

    # Preprocess
    X = preprocessor.transform(df)

    # Predict probability 
    p_non_renew = model.predict_proba(X)[:, 1][0]
    p_renew = 1 - p_non_renew

    # Optimize incentive
    result = minimize_scalar(
        net_revenue,
        bounds=(0, 2000),
        args=(p_renew, data['premium']),
        method='bounded'
    )

    return {
        "renewal_probability": round(float(p_renew), 4),
        "optimal_incentive": round(float(result.x), 4)
    }
