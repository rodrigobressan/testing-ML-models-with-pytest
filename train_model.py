from sklearn.linear_model import LinearRegression
from sklearn.externals import joblib
import numpy as np


def train_dummy_model():
    """
    Used to train our dummy model with a simple set of features. In this example, the output for each feature will
    be calculated as input * 10. e.g.:

    Input 1 -> Output 10
    Input 10 -> Output 100
    Input 30 -> Output 300
    :return:
    """
    print('Training dummy model')

    X_train = np.array([1, 3, 5, 7, 9]).reshape(-1, 1)
    y_train = np.array([10, 30, 50, 70, 90]).reshape(-1, 1)

    # train our model
    model = LinearRegression()
    model.fit(X_train, y_train)

    joblib.dump(model, 'trained_dummy_model.sav')


if __name__ == '__main__':
    train_dummy_model()
