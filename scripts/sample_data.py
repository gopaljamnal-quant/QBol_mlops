import numpy as np
import pandas as pd
from src.config import settings
from src.storage import write_csv_to_blob

def generate_and_upload(n_rows: int = 1000):
    rng = np.random.default_rng(42)
    tenure = rng.integers(1, 72, size=n_rows)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], p=[0.55, 0.25, 0.20], size=n_rows)
    tech_support = rng.choice(["Yes", "No"], p=[0.35, 0.65], size=n_rows)
    monthly = np.round(rng.uniform(25.0, 120.0, size=n_rows), 2)
    tickets = rng.poisson(1.5, size=n_rows)

    # Realistic churn probability formula
    logit = (
        -1.5
        + 1.4 * (contract == "Month-to-month")
        - 0.03 * tenure
        + 0.015 * monthly
        + 0.45 * tickets
        + 0.6 * (tech_support == "No")
    )
    prob = 1 / (1 + np.exp(-logit))
    churn = rng.binomial(1, prob)

    df = pd.DataFrame({
        "customer_id": [f"CUST-{10000+i}" for i in range(n_rows)],
        "tenure_months": tenure,
        "monthly_charges": monthly,
        "total_charges": np.round(monthly * tenure, 2),
        "contract_type": contract,
        "tech_support": tech_support,
        "support_tickets_last_30d": tickets,
        "churn": churn
    })

    write_csv_to_blob(df, settings.RAW_DATA_BLOB)
    print(f"Uploaded {n_rows} synthetic customer records to {settings.AZURE_CONTAINER_NAME}/{settings.RAW_DATA_BLOB}")

if __name__ == "__main__":
    generate_and_upload()
