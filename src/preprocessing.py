import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def prepare_features(df: pd.DataFrame):
    """
    Prepare features and target variable.
    """
    X = df.drop(columns=["delayed"])
    y = df["delayed"]

    return X, y


def build_preprocessing_pipeline():
    """
    Create preprocessing pipeline for numeric and categorical features.
    """
    numeric_features = [
        "DISTANCE",
        "dep_hour",
        "DAY_OF_WEEK",
        "MONTH"
    ]

    categorical_features = [
        "OP_CARRIER",
        "ORIGIN",
        "DEST"
    ]

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features)
        ]
    )

    return preprocessor


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into train and test sets.
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
