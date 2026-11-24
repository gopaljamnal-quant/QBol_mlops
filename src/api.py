import pandas as pd
import numpy as np
import shap
import mlflow
import mlflow.sklearn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal, List
from src.config import settings
from src.storage import append_inference_log

app = FastAPI(title="Customer Churn Prediction API", version="1.0.0")

class CustomerFeatures(BaseModel):
    customer_id: str
    tenure_months: int = Field(ge=0, le=120)
    monthly_charges: float = Field(gt=0, le=500)
    total_charges: float | None = Field(default=None, ge=0)
    contract_type: Literal["Month-to-month", "One year", "Two year"]
    tech_support: Literal["Yes", "No"]
    support_tickets_last_30d: int = Field(ge=0, le=50)

class RiskDriver(BaseModel):
    feature: str
    shap_impact: float

class PredictionResponse(BaseModel):
    customer_id: str
    churn_probability: float
    risk_tier: str
    top_drivers: List[RiskDriver]

# Load @champion pipeline at startup
mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
MODEL_URI = f"models:/{settings.REGISTERED_MODEL_NAME}@champion"
pipeline = mlflow.sklearn.load_model(MODEL_URI)

# Extract LightGBM classifier & preprocessor for fast TreeSHAP explanations
feature_eng = pipeline.named_steps["feature_eng"]
preprocessor = pipeline.named_steps["preprocessor"]
classifier = pipeline.named_steps["classifier"]
explainer = shap.TreeExplainer(classifier)

@app.get("/health")
def health():
    return {"status": "healthy", "model_uri": MODEL_URI}

@app.post("/predict", response_model=PredictionResponse)
def predict_churn(customer: CustomerFeatures):
    try:
        input_df = pd.DataFrame([customer.model_dump(exclude={"customer_id"})])
        prob = float(pipeline.predict_proba(input_df)[0, 1])

        # Compute SHAP values for local explainability
        engineered_df = feature_eng.transform(input_df)
        transformed_array = preprocessor.transform(engineered_df)
        feature_names = preprocessor.get_feature_names_out()

        shap_vals = explainer.shap_values(transformed_array)
        # LightGBM binary returns list of [class_0, class_1] or 2D array depending on version
        class1_shap = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]

        top_indices = np.argsort(np.abs(class1_shap))[::-1][:3]
        drivers = [
            RiskDriver(
                feature=feature_names[i].replace("num__", "").replace("cat__", ""),
                shap_impact=round(float(class1_shap[i]), 4)
            )
            for i in top_indices
        ]

        risk_tier = "HIGH" if prob >= 0.70 else ("MEDIUM" if prob >= 0.40 else "LOW")

        # Async-safe logging to Azure Blob for drift monitoring
        log_row = input_df.copy()
        log_row["churn_probability"] = prob
        append_inference_log(log_row)

        return PredictionResponse(
            customer_id=customer.customer_id,
            churn_probability=round(prob, 4),
            risk_tier=risk_tier,
            top_drivers=drivers
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
