import pandas as pd
from sklearn.cluster import KMeans,DBSCAN
from sklearn.mixture import GaussianMixture
import numpy as np
from sklearn.metrics import accuracy_score,confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import sys
import os
import preproccesing, transformation
from pipelineObj import Pipeline
import mlflow.sklearn
import mlflow
import mlflow.models
import warnings
from mlflow.models import infer_signature
import itertools
from urllib.parse import urlparse



if __name__ == '__main__':
    mlflow.set_tracking_uri('http://localhost:5001')
    experiment = mlflow.set_experiment("clustering-titles")
    print("mlflow tracking uri:", mlflow.get_tracking_uri())
    print("experiment:", experiment)

    # Predecir y calcular la matriz de confusión
    path = '../../data/raw/pricerunner_aggregate.csv'
    pipe = Pipeline()
    n_clusters = 10
    random_state = 2
    tp,fp,fn,tn = pipe.run_pipeline(path,KMeans(n_clusters=n_clusters,random_state=random_state))

    mlflow.log_param('n_clusters', n_clusters)
    mlflow.log_param('random_state', random_state)

    mlflow.log_metric("tp", np.sum(tp))
    mlflow.log_metric("tn", np.sum(tn))
    mlflow.log_metric("fp", np.sum(fp))
    mlflow.log_metric("fn", np.sum(fn))

    # Guardar modelo en MLflow
    model_info = mlflow.sklearn.log_model(pipe.modelo, artifact_path=f"Kmeans_model")
    print(f"Saved Kmeans model:", model_info.model_uri)