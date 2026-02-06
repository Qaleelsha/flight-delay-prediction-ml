from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def train_logistic_regression(preprocessor):
    """
    Train Logistic Regression Model.
    """

    model = Pipeline(steps =[
        ("preprocessing", preprocessor),
        ("classifier", LogisticRegression(
            max_iter = 1000,
            class_weight = "balanced",
            random_state = 42
        )),
    ])
    return model

def train_random_forest(preprocessor):
    """
    Train Random Forest Classifier.
    """

    model = Pipeline(steps =[
        ("preprocessing", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators = 200,
            max_depth = 12,
            random_state = 42,
            n_jobs = -1
        ))
    ])
    return model

def train_gradient_boosting(preprocessor):
    """
    Train Gradient Boosting.
    """

    model = Pipeline(steps =[
        ("preprocessing", preprocessor),
        ("classifier", GradientBoostingClassifier(
            n_estimators = 150,
            learning_rate = 0.1,
            max_depth = 3,
            random_state = 42
        ))
    ])
    return model