import time
import numpy as np
import requests
from src.monitor import check_drift

API_URL = "http://localhost:8000/predict"

def simulate_production_drift(n_requests: int = 50):
    rng = np.random.default_rng(99)
    print(f"Sending {n_requests} drifted customer requests to {API_URL}...")

    for i in range(n_requests):
        # Simulate sudden business shift: higher prices, shorter tenure, higher ticket volume
        payload = {
            "customer_id": f"DRIFT-{20000 + i}",
            "tenure_months": int(rng.integers(1, 12)),                     # Shifted down from 1-72
            "monthly_charges": round(float(rng.uniform(95.0, 180.0)), 2),  # Shifted up from 25-120
            "total_charges": None,
            "contract_type": str(rng.choice(["Month-to-month", "One year"], p=[0.90, 0.10])),
            "tech_support": "No",
            "support_tickets_last_30d": int(rng.poisson(5.0))              # Shifted up from 1.5
        }

        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            data = response.json()
            if (i + 1) % 10 == 0:
                print(
                    f"[{i+1}/{n_requests}] Sample Prediction -> "
                    f"Prob: {data['churn_probability']} ({data['risk_tier']}) | "
                    f"Top Driver: {data['top_drivers']['feature']}"
                )
        else:
            print(f"[ERROR] Status {response.status_code}: {response.text}")

        time.sleep(0.05)

    print("\n--- Running Evidently AI Drift Detection ---")
    check_drift()

if __name__ == "__main__":
    simulate_production_drift()
