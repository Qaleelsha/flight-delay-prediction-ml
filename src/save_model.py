import joblib
from pathlib import Path


def save_model(model, model_path: str):
    """
    Save trained model pipeline to disk.
    """
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)


def load_model(model_path: str):
    """
    Load trained model pipeline from disk.
    """
    return joblib.load(model_path)
