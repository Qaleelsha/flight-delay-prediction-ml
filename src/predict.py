import pandas as pd


def make_prediction(model, input_data: dict):
    """
    Make prediction for a single flight input.
    """
    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    return prediction, probability
