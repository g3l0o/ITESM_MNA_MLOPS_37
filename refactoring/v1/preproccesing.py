
import pandas as pd
import re
from nltk.corpus import stopwords
import nltk
from datetime import datetime
import os

def preprocces(df:pd.DataFrame):
    nltk.download('stopwords')

    rename_mapper = {column:column.strip() for column in df.columns}
    new_df = df.rename(rename_mapper,axis=1)

    cluster_ids = new_df['Category ID'].unique()
    label_mapper = {cluster_ids[i]:i for i in range(len(cluster_ids))}
    new_df['Category ID'] = new_df['Category ID'].map(label_mapper)


    new_df.drop_duplicates(inplace=True)
    new_df.dropna(inplace=True)
    new_df.drop(['Product ID','Merchant ID','Cluster ID', 'Cluster Label'],axis=1,inplace=True)

    strings = new_df['Product Title'].values

    new_df.drop('Product Title',axis=1,inplace=True)

    onlyLetters = r'[^A-Za-z]'
    strings = [re.sub(onlyLetters,' ',string) for string in strings]

    only2LenWords = r'\b\w{2,}\b'
    stringsList= [re.findall(only2LenWords,string) for string in strings]



    noStopWords = []
    for string in stringsList:
        noStopWords.append([word for word in string if word not in stopwords.words('english')])




    new_df['words'] = noStopWords
    time = ''.join(str(datetime.now())[:-7])
    current_dir = os.path.dirname(__file__)
    data_path = os.path.join(current_dir, '../../data/processed/')
    new_df.to_csv(f'{data_path + time}.csv',index=False)

    return new_df
