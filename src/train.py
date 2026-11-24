import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score, fbeta_score
from src.config import settings
from src.storage import read_csv_from_blob, write_csv_to_blob
from src.validate_data import validate_dataset
from src.features import build_training_pipeline

def get_champion_pr_auc(client: MlflowClient) -> float:
    try:
        version_info = client.get_model_version_by_alias(settings.REGISTERED_MODEL_NAME, "champion")
        run = client.get_run(version_info.run_id)
        return float(run.data.metrics.get("pr_auc", 0.0))
    except Exception:
        return 0.0

def run_training():
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(settings.EXPERIMENT_NAME)
    client = MlflowClient()

    # 1. Load & Validate from Azure Blob Storage
    raw_df = read_csv_from_blob(settings.RAW_DATA_BLOB)
    clean_df = validate_dataset(raw_df)

    X = clean_df.drop(columns=["customer_id", "churn"])
    y = clean_df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Save reference baseline in Azure Blob for Evidently drift detection
    ref_df = X_test.copy()
    ref_df["churn"] = y_test.values
    write_csv_to_blob(ref_df, settings.REFERENCE_DATA_BLOB)

    params = {
        "n_estimators": 150,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "class_weight": "balanced",
        "random_state": 42,
        "verbose": -1
    }

    with mlflow.start_run() as run:
        pipeline = build_training_pipeline(params)
        pipeline.fit(X_train, y_train)

        y_probs = pipeline.predict_proba(X_test)[:, 1]
        y_preds = (y_probs >= 0.5).astype(int)

        pr_auc = average_precision_score(y_test, y_probs)
        roc_auc = roc_auc_score(y_test, y_probs)
        f2 = fbeta_score(y_test, y_preds, beta=2)

        mlflow.log_params(params)
        mlflow.log_metrics({"pr_auc": pr_auc, "roc_auc": roc_auc, "f2_score": f2})

        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name=settings.REGISTERED_MODEL_NAME
        )

        # 2. Champion / Challenger Promotion Gate
        champion_pr_auc = get_champion_pr_auc(client)
        new_version = model_info.registered_model_version

        print(f"Challenger PR-AUC: {pr_auc:.4f} | Current Champion PR-AUC: {champion_pr_auc:.4f}")

        if pr_auc >= champion_pr_auc:
            client.set_registered_model_alias(
                name=settings.REGISTERED_MODEL_NAME,
                alias="champion",
                version=new_version
            )
            print(f"[PROMOTED] Model version {new_version} set as @champion.")
        else:
            print(f"[REJECTED] Version {new_version} did not beat @champion.")

if __name__ == "__main__":
    run_training()
