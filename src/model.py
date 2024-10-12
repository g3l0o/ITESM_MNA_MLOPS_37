import sys
from sklearn.linear_model import LogisticRegression
import pandas as pd
import joblib


def model(x_train_path: str, y_train_path: str):
    x_train = pd.read_csv(x_train_path)
    y_train = pd.read_csv(y_train_path)
    model = LogisticRegression(max_iter=1000)
    model.fit(X=x_train, y=y_train)
    return model


if __name__ == '__main__':
    x_train_path = sys.argv[1]
    y_train_path = sys.argv[2]
    model_path = sys.argv[3]
    model = model(x_train_path, y_train_path)
    joblib.dump(model, model_path)