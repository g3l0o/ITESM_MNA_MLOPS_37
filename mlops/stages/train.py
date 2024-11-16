import pandas as pd
import yaml
import argparse
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,accuracy_score,recall_score,confusion_matrix
import joblib
from mlflow.models import infer_signature
import mlflow.sklearn
import mlflow
import mlflow.models
import numpy as np

def obtain_estimators():
    return{
        'k_means':KMeans,
        'r_forest':RandomForestClassifier
    }


def train_model(config_path):
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    mlflow.set_tracking_uri(config['server']['url'])
    experiment = mlflow.set_experiment(config['server']['experiment_name'])

    estimator_name = config['train']['estimator_name']
    estimators= obtain_estimators()
    estimator = estimators[estimator_name]()
    params = config['train']['estimators'][estimator_name]['param_grid']
    cv = config['train']['cv']
    train_df = pd.read_csv(config['transform_data']['train_embedds_dir'])
    x = train_df.iloc[:,:-1]
    y = train_df.iloc[:,-1]

    model = GridSearchCV(estimator=estimator,
                        param_grid=params,
                        scoring="f1_weighted",
                        cv=cv
    )


    model.fit(x,y)
    best_params = model.best_params_
    predictions = model.predict(x)
    for param, value in best_params.items():
        mlflow.log_param(param, value)

    accuracy = accuracy_score(y, predictions)
    recall = recall_score(y, predictions, average='weighted')
    f1 = f1_score(y, predictions, average='weighted')

    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_metric('recall', recall)
    mlflow.log_metric('f1_score', f1)


    models_path = config['train']['model_path']
    joblib.dump(model, models_path)
    model_info = mlflow.sklearn.log_model(model, artifact_path=f"r_forest")
if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    train_model(config_path=args.config)