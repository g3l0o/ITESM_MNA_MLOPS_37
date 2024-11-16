import pandas as pd
import yaml 
import argparse
import joblib
from sklearn.metrics import accuracy_score,recall_score,f1_score
import json
from sklearn.metrics import ConfusionMatrixDisplay,confusion_matrix
import matplotlib.pyplot as plt

def evaluate(config_path):
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)
    model = joblib.load(config['train']['model_path'])
    test_df = pd.read_csv(config['transform_data']['test_embedds_dir'])
    x = test_df.iloc[:,:-1]
    y = test_df.iloc[:,-1]
    predictions = model.predict(x)
    ac = accuracy_score(y,predictions)
    rec = recall_score(y,predictions,average='weighted')
    f1Sc = f1_score(y,predictions,average='weighted')

    metrics = {'accuracy':ac,
            'recall':rec,
            'f1_score':f1Sc}
    
    file_name = config['evaluate']['metrics_dir']
    json.dump(
        obj=metrics,
        fp=open(file_name,'w')
    )
    cm = confusion_matrix(y,predictions,labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                display_labels=model.classes_)
    disp.plot()
    plt.show
    plt.savefig(config['evaluate']['cm_path'])
    plt.close()


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    evaluate(config_path=args.config)