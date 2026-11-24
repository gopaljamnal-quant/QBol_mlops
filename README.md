# End-to-End Closed-Loop MLOps Pipeline: Customer Churn Prediction

An automated, production-grade Machine Learning Operations (MLOps) system built on **Microsoft Azure**. It trains an explainable **LightGBM** churn model with **SHAP** attribution, enforces **Pandera** data contracts, manages **Champion/Challenger** model promotion in **MLflow**, serves low-latency predictions via **FastAPI + Docker**, and automatically triggers retraining via **Evidently AI** and **GitHub Actions** when production data drift occurs.

---

## Architecture Overview

```text
[Azure Blob Storage] ──► [Pandera Data Quality Gate] ──► [LightGBM + Scikit-Learn Pipeline]
                                                                       │
[GitHub Actions CI/CD] ◄── [Evidently AI Drift Monitor]        [MLflow Model Registry]
         │                             ▲                         (@champion vs @challenger)
         ▼                             │                               │
[Azure Container Apps] ◄─────── [FastAPI + TreeSHAP] ◄─────────────────┘
```

### Key Engineering Highlights
- **Zero-Key Cloud Security:** Authenticates with Azure Blob Storage using `DefaultAzureCredential` (Azure CLI locally, Managed Identity in production)—zero hardcoded storage keys.
- **Zero Training-Serving Skew:** Feature engineering (`RatioFeatureEngineer`), imputation, encoding, and `LGBMClassifier` are serialized as a single `scikit-learn` pipeline artifact in MLflow.
- **Imbalanced Metric Optimization:** Evaluates models on **PR-AUC** and **F2-Score** rather than raw accuracy, promoting a `@challenger` model to `@champion` only when it outperforms production baselines.
- **Real-Time Explainability (SHAP):** Every `/predict` call returns both the churn probability and the **Top 3 SHAP risk drivers** so retention teams understand *why* a customer is at risk.
- **Closed-Loop Drift Retraining:** Live predictions are logged back to Azure Blob Storage. **Evidently AI** runs statistical drift tests (PSI / Wasserstein) and fires a GitHub Actions `repository_dispatch` webhook to retrain when drift crosses `30%`.

---

## Repository Structure

```text
mlops-customer-churn/
├── .github/workflows/
│   └── retrain_and_deploy.yml   # Automated CI/CD & drift-triggered retraining
├── scripts/
│   ├── generate_sample_data.py  # Seeds synthetic customer data into Azure Blob
│   └── simulate_drift.py        # Simulates production drift & triggers monitor
├── src/
│   ├── config.py                # Pydantic environment configuration
│   ├── storage.py               # Keyless Azure Blob Storage I/O wrapper
│   ├── validate_data.py         # Pandera schema & target distribution checks
│   ├── features.py              # Leak-free feature engineering & preprocessing
│   ├── train.py                 # Training, MLflow tracking & @champion promotion
│   ├── monitor.py               # Evidently AI drift detection + webhook trigger
│   └── api.py                   # FastAPI inference service with TreeSHAP
├── Dockerfile                   # Non-root Linux slim container
├── requirements.txt             # Pinned Python dependencies
└── README.md
```

---

## Quickstart Guide (Local VS Code + Azure Blob Storage)

### 1. Provision Azure Storage & Assign RBAC
```bash
az login
az group create --name rg-churn-mlops --location uksouth
az storage account create --name <uniquestoragename> --resource-group rg-churn-mlops --sku Standard_LRS

USER_ID=\$(az ad signed-in-user show --query id -o tsv)
STORAGE_ID=\$(az storage account show --name <uniquestoragename> --resource-group rg-churn-mlops --query id -o tsv)
az role assignment create --assignee \$USER_ID --role "Storage Blob Data Contributor" --scope \$STORAGE_ID
```

### 2. Install Dependencies & Configure Environment
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the root directory:
```env
AZURE_STORAGE_ACCOUNT_URL=https://<uniquestoragename>.blob.core.windows.net
AZURE_CONTAINER_NAME=churn-mlops
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
DRIFT_SHARE_THRESHOLD=0.30
```

### 3. Run the End-to-End Pipeline
```bash
# Step 1: Generate and upload baseline customer dataset to Azure Blob Storage
python -m scripts.generate_sample_data

# Step 2: Validate data, train LightGBM, and promote @champion in MLflow
python -m src.train

# Step 3: Start the FastAPI inference server
uvicorn src.api:app --reload --port 8000
```
Visit `http://localhost:8000/docs` to test `POST /predict` interactively.

### 4. Simulate Production Drift & Automated Detection
In a second terminal (while `uvicorn` is running):
```bash
python -m scripts.simulate_drift
```
This sends 50 drifted customer profiles to `/predict`, logs them to Azure Blob Storage, runs **Evidently AI**, and triggers the drift alert.

---

## Cloud Deployment (Azure Container Apps)
```bash
az acr build --registry <your-acr-name> --image churn-api:v1 .

az containerapp create \
  --name churn-prediction-api \
  --resource-group rg-churn-mlops \
  --environment <your-container-env> \
  --image <your-acr-name>.azurecr.io/churn-api:v1 \
  --target-port 8000 \
  --ingress external \
  --system-assigned \
  --env-vars \
    AZURE_STORAGE_ACCOUNT_URL="https://<uniquestoragename>.blob.core.windows.net" \
    AZURE_CONTAINER_NAME="churn-mlops"
```
*(Remember to assign the `Storage Blob Data Contributor` role to your Container App's System-Assigned Managed Identity.)*
