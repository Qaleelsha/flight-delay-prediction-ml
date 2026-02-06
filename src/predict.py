import pandas as pd

def make_prediction(model, input_data: dict, threshold: float = 0.35):
    """
    Make prediction for a single flight input.
    """
    input_df = pd.DataFrame([input_data])

    probability = model.predict_proba(input_df)[0][1]
    prediction = int(probability >= threshold)

    return prediction, probability
