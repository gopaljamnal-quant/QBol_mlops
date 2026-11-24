import requests
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from src.config import settings
from src.storage import read_csv_from_blob

def trigger_github_retrain():
    if not settings.GITHUB_REPO or not settings.GITHUB_TOKEN:
        print("[WARN] GitHub credentials not set. Skipping remote webhook trigger.")
        return

    url = f"https://github.com{settings.GITHUB_REPO}/dispatches"
    headers = {
        "Authorization": f"Bearer {settings.GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json"
    }
    response = requests.post(url, json={"event_type": "data-drift-detected"}, headers=headers)
    print(f"[WEBHOOK] GitHub Actions triggered. Status: {response.status_code}")

def check_drift():
    ref_df = read_csv_from_blob(settings.REFERENCE_DATA_BLOB).drop(columns=["churn"], errors="ignore")
    prod_df = read_csv_from_blob(settings.INFERENCE_LOGS_BLOB).drop(columns=["churn_probability"], errors="ignore")

    if len(prod_df) < 30:
        print(f"Only {len(prod_df)} production rows logged. Need at least 30 for reliable drift stats.")
        return

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_df, current_data=prod_df)
    result = report.as_dict()

    drift_metrics = result["metrics"][0]["result"]
    drift_share = drift_metrics["share_of_drifted_columns"]
    dataset_drifted = drift_metrics["dataset_drift"]

    print(f"Drifted Feature Share: {drift_share:.2%} (Threshold: {settings.DRIFT_SHARE_THRESHOLD:.2%})")

    if dataset_drifted or drift_share >= settings.DRIFT_SHARE_THRESHOLD:
        print("[ALERT] Significant data drift detected! Triggering retraining pipeline...")
        trigger_github_retrain()
    else:
        print("[OK] Data distribution is stable.")

if __name__ == "__main__":
    check_drift()
