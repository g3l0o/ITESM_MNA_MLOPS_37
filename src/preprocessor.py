import sys

import pandas as pd
import re
from nltk.corpus import stopwords
import nltk
from datetime import datetime
import os


def preprocces(input_filepath: str, output_filepath: str) -> pd.DataFrame:

    df = pd.read_csv(input_filepath)

    nltk.download('stopwords')

    rename_mapper = {column:column.strip() for column in df.columns}
    new_df = df.rename(rename_mapper,axis=1)

    cluster_ids = new_df['category_id'].unique()
    label_mapper = {cluster_ids[i]:i for i in range(len(cluster_ids))}
    new_df['category_id'] = new_df['category_id'].map(label_mapper)


    new_df.drop_duplicates(inplace=True)
    new_df.dropna(inplace=True)
    new_df.drop(['product_id','merchant_id','cluster_id', 'cluster_label'],axis=1,inplace=True)

    strings = new_df['product_title'].values

    new_df.drop('product_title',axis=1,inplace=True)

    onlyLetters = r'[^A-Za-z]'
    strings = [re.sub(onlyLetters,' ',string) for string in strings]

    only2LenWords = r'\b\w{2,}\b'
    stringsList= [re.findall(only2LenWords,string) for string in strings]
    noStopWords = []
    for string in stringsList:
        noStopWords.append([word for word in string if word not in stopwords.words('english')])

    new_df['words'] = noStopWords
    new_df.to_csv(output_filepath,index=False)

    return new_df


if __name__ == '__main__':
    input_file_path = sys.argv[1]
    output_file = sys.argv[2]
    data = preprocces(input_file_path, output_file)
