from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Azure Blob Storage Settings
    AZURE_STORAGE_ACCOUNT_URL: str  # e.g., https://<youraccount>.blob.core.windows.net
    AZURE_CONTAINER_NAME: str = "churn-mlops"
    RAW_DATA_BLOB: str = "data/raw/customer_churn.csv"
    REFERENCE_DATA_BLOB: str = "data/reference/reference_churn.csv"
    INFERENCE_LOGS_BLOB: str = "data/logs/inference_logs.csv"

    # MLflow Tracking & Model Registry (Local file store or DagsHub/AzureML URI)
    MLFLOW_TRACKING_URI: str = "sqlite:///mlflow.db"
    EXPERIMENT_NAME: str = "customer-churn-lgbm"
    REGISTERED_MODEL_NAME: str = "churn-predictor"

    # Monitoring & GitHub Actions Webhook (Optional for automated retraining)
    DRIFT_SHARE_THRESHOLD: float = 0.30
    GITHUB_REPO: str = ""   # e.g., "username/mlops-customer-churn"
    GITHUB_TOKEN: str = ""  # Personal Access Token with repo scope

    class Config:
        env_file = ".env"

settings = Settings()
