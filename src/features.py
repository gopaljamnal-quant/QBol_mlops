import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from lightgbm import LGBMClassifier

NUMERIC_FEATURES = ["tenure_months", "monthly_charges", "total_charges", "support_tickets_last_30d", "charge_ratio"]
CATEGORICAL_FEATURES = ["contract_type", "tech_support"]

class RatioFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        df["total_charges"] = df["total_charges"].fillna(df["monthly_charges"] * df["tenure_months"])
        df["charge_ratio"] = df["monthly_charges"] / (df["tenure_months"] + 1.0)
        return df

def build_training_pipeline(params: dict) -> Pipeline:
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    model = LGBMClassifier(**params)

    return Pipeline(steps=[
        ("feature_eng", RatioFeatureEngineer()),
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])
