import pytest
import pandas as pd
import numpy as np
import pandera as pa
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from src.validate_data import validate_dataset
from src.features import build_training_pipeline, RatioFeatureEngineer

# --- Fixtures ---
@pytest.fixture
def sample_valid_df() -> pd.DataFrame:
    return pd.DataFrame({
        "customer_id": [f"CUST-{i}" for i in range(20)],
        "tenure_months": [2, 24, 48, 5, 60] * 4,
        "monthly_charges": [95.0, 55.0, 40.0, 110.0, 35.0] * 4,
        "total_charges": [190.0, 1320.0, 1920.0, np.nan, 2100.0] * 4,
        "contract_type": ["Month-to-month", "One year", "Two year", "Month-to-month", "Two year"] * 4,
        "tech_support": ["No", "Yes", "Yes", "No", "Yes"] * 4,
        "support_tickets_last_30d": [4, 1, 0, 5, 0] * 4,
        "churn": [1, 0, 0, 1, 0] * 4
    })

# --- 1. Data Validation Tests (Pandera) ---
def test_validate_dataset_passes_valid_data(sample_valid_df):
    result = validate_dataset(sample_valid_df)
    assert len(result) == 20
    assert list(result.columns) == list(sample_valid_df.columns)

def test_validate_dataset_blocks_invalid_schema(sample_valid_df):
    bad_df = sample_valid_df.copy()
    bad_df.loc[0, "tenure_months"] = -5  # Violates Check.ge(0)
    bad_df.loc[1, "contract_type"] = "Lifetime"  # Violates Check.isin(...)

    with pytest.raises(pa.errors.SchemaErrors):
        validate_dataset(bad_df)

def test_validate_dataset_blocks_anomalous_churn_rate(sample_valid_df):
    all_zero_df = sample_valid_df.copy()
    all_zero_df["churn"] = 0  # 0% churn violates the 2%-80% sanity check

    with pytest.raises(ValueError, match="Anomalous target churn rate"):
        validate_dataset(all_zero_df)

# --- 2. Feature Engineering & Pipeline Tests ---
def test_ratio_feature_engineer_imputes_total_charges(sample_valid_df):
    engineer = RatioFeatureEngineer()
    X = sample_valid_df.drop(columns=["customer_id", "churn"])
    transformed = engineer.transform(X)

    assert "charge_ratio" in transformed.columns
    assert not transformed["total_charges"].isna().any()
    # Row 3 had total_charges=NaN, tenure=5, monthly=110.0 -> expected 550.0
    assert transformed.loc[3, "total_charges"] == pytest.approx(550.0)

def test_training_pipeline_end_to_end(sample_valid_df):
    X = sample_valid_df.drop(columns=["customer_id", "churn"])
    y = sample_valid_df["churn"]

    pipeline = build_training_pipeline({"n_estimators": 10, "random_state": 42, "verbose": -1})
    pipeline.fit(X, y)

    probs = pipeline.predict_proba(X)[:, 1]
    assert len(probs) == len(X)
    assert np.all((probs >= 0.0) & (probs <= 1.0))

# --- 3. FastAPI & SHAP Serving Tests (Mocked MLflow & Azure Blob) ---
@pytest.fixture
def mock_api_client(sample_valid_df):
    # Train a tiny real pipeline in memory so TreeSHAP works genuinely during the test
    X = sample_valid_df.drop(columns=["customer_id", "churn"])
    y = sample_valid_df["churn"]
    real_tiny_pipeline = build_training_pipeline({"n_estimators": 10, "random_state": 42, "verbose": -1})
    real_tiny_pipeline.fit(X, y)

    with patch.dict("os.environ", {"AZURE_STORAGE_ACCOUNT_URL": "https://windows.net"}):
        with patch("mlflow.sklearn.load_model", return_value=real_tiny_pipeline):
            with patch("src.api.append_inference_log") as mock_log:
                import importlib
                import src.api
                importlib.reload(src.api)
                client = TestClient(src.api.app)
                yield client, mock_log

def test_api_health_endpoint(mock_api_client):
    client, _ = mock_api_client
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_api_predict_returns_probability_and_shap_drivers(mock_api_client):
    client, mock_log = mock_api_client
    payload = {
        "customer_id": "CUST-9999",
        "tenure_months": 3,
        "monthly_charges": 105.50,
        "total_charges": None,
        "contract_type": "Month-to-month",
        "tech_support": "No",
        "support_tickets_last_30d": 4
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    body = response.json()
    assert body["customer_id"] == "CUST-9999"
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["risk_tier"] in ["LOW", "MEDIUM", "HIGH"]
    assert len(body["top_drivers"]) == 3
    assert "feature" in body["top_drivers"]
    assert "shap_impact" in body["top_drivers"]
    mock_log.assert_called_once()

def test_api_predict_rejects_invalid_payload(mock_api_client):
    client, _ = mock_api_client
    invalid_payload = {
        "customer_id": "CUST-BAD",
        "tenure_months": -10,            # Invalid negative tenure
        "monthly_charges": 50.0,
        "contract_type": "InvalidPlan",  # Invalid category
        "tech_support": "No",
        "support_tickets_last_30d": 1
    }

    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422
