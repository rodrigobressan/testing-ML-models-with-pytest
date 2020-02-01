from sklearn.externals import joblib
from sklearn.metrics import r2_score
import numpy as np


def test_prediction_positive_values():
    """
    Used to test positive values with our trained model
    :return:
    """
    model = joblib.load('trained_dummy_model.sav')

    X_test = np.array([2, 4, 6, 8, 10, 50]).reshape(-1, 1)
    y_test = np.array([20, 40, 60, 80, 100, 500]).reshape(-1, 1)

    y_preds = model.predict(X_test)
    r2 = r2_score(y_test, y_preds)

    assert r2 == 1


def test_prediction_negative_values():
    """
    Used to test negative values with our trained model
    :return:
    """
    model = joblib.load('trained_dummy_model.sav')

    X_test = np.array([-2, -4, -6, -8, -10, -50]).reshape(-1, 1)
    y_test = np.array([-20, -40, -60, -80, -100, -500]).reshape(-1, 1)

    y_preds = model.predict(X_test)
    r2 = r2_score(y_test, y_preds)

    assert r2 == 1
