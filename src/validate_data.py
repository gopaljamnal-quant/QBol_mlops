import pandas as pd
import pandera as pa
from pandera import Column, Check, DataFrameSchema

churn_schema = DataFrameSchema(
    columns={
        "customer_id": Column(str, nullable=False, unique=True),
        "tenure_months": Column(int, checks=[Check.ge(0), Check.le(120)], nullable=False),
        "monthly_charges": Column(float, checks=[Check.gt(0), Check.le(500)], nullable=False),
        "total_charges": Column(float, checks=[Check.ge(0)], nullable=True),
        "contract_type": Column(
            str,
            checks=Check.isin(["Month-to-month", "One year", "Two year"]),
            nullable=False
        ),
        "tech_support": Column(str, checks=Check.isin(["Yes", "No"]), nullable=False),
        "support_tickets_last_30d": Column(int, checks=[Check.ge(0), Check.le(50)], nullable=False),
        "churn": Column(int, checks=Check.isin([0, 1]), nullable=False),
    },
    strict=False
)

def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    validated_df = churn_schema.validate(df, lazy=True)
    churn_rate = validated_df["churn"].mean()
    if not (0.02 <= churn_rate <= 0.80):
        raise ValueError(f"Anomalous target churn rate detected: {churn_rate:.2%}")
    print(f"[VALIDATION PASSED] {len(validated_df)} rows verified. Churn rate: {churn_rate:.2%}")
    return validated_df
