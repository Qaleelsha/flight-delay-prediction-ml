import pandas as pd


def load_flight_data(file_path: str) -> pd.DataFrame:
    """
    Load flight data from CSV file.
    """
    df = pd.read_csv(file_path)
    return df


def create_delay_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary target variable:
    1 -> Arrival delay >= 15 minutes
    0 -> Arrival delay < 15 minutes
    """
    df["delayed"] = (df["ARR_DELAY"] >= 15).astype(int)
    return df


def remove_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove columns that are not available before departure
    or cause data leakage.
    """
    columns_to_drop = [
        "ARR_DELAY",
        "DEP_DELAY",
        "ACTUAL_ELAPSED_TIME",
        "AIR_TIME",
        "WHEELS_OFF",
        "WHEELS_ON",
        "TAXI_IN",
        "TAXI_OUT",
        "CANCELLED",
        "DIVERTED"
    ]

    existing_columns = [col for col in columns_to_drop if col in df.columns]
    df = df.drop(columns=existing_columns)

    return df
