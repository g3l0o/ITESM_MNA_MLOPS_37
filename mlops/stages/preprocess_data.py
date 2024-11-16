import pandas as pd
import re
from nltk.corpus import stopwords
import nltk
from datetime import datetime
import os
import yaml
import argparse

def preprocces(config_path):
    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)

    nltk.download('stopwords')
    df = pd.read_csv(config['load_data']['loaded_data_dir'])

    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    df.drop(['Product ID','Merchant ID','Cluster ID', 'Cluster Label'],axis=1,inplace=True)

    strings = df['Product Title'].values

    df.drop('Product Title',axis=1,inplace=True)

    onlyLetters = r'[^A-Za-z]'
    strings = [re.sub(onlyLetters,' ',string) for string in strings]

    only2LenWords = r'\b\w{2,}\b'
    stringsList= [re.findall(only2LenWords,string) for string in strings]



    noStopWords = []
    for string in stringsList:
        noStopWords.append([word for word in string if word not in stopwords.words('english')])


    df['words'] = noStopWords
    df.to_csv(config['preprocess_data']['processed_data_dir'],index=False)

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    preprocces(config_path=args.config)

