import joblib
import pandas as pd
import sys
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def evaluator (model_path, x_test_path, y_test_path):
    model = joblib.load(model_path)
    x_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path)

    predictions = model.predict(X=x_test)
    acc = accuracy_score(y_test, predictions)
    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test, predictions))
    print(acc)


if __name__ == '__main__':
    model_path = sys.argv[1]
    X_test_path = sys.argv[2]
    y_test_path = sys.argv[3]