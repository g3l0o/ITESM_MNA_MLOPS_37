import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
import fasttext
import fasttext.util
import numpy as np
import yaml
import argparse
import ast


def transform_data(config_path):
    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)

    test_size = config['transform_data']['test_size']
    random_state = config['base']['random_state']
    df = pd.read_csv(config['preprocess_data']['processed_data_dir'])
    df['words'] = df['words'].apply(ast.literal_eval)

    X = df['words'].values
    y = df['Category ID'].values


    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_size,random_state=random_state)
    diccionario = Counter()
    for sentence in X_train:
        diccionario.update(sentence)
    min_freq = 2
    mi_dicc = {key: val for key, val in diccionario.items() if val >= min_freq}


    train_vocab = []
    for sentence in X_train:
        train_vocab.append([word for word in sentence if word in mi_dicc])
    test_vocab = []
    for sentence in X_test:
        test_vocab.append([word for word in sentence if word in mi_dicc])

    embedd_model_path = config['transform_data']['embedd_model_dir']
    ft_model = fasttext.load_model(embedd_model_path)

    vocabs = [train_vocab,test_vocab]
    train_vocab = pd.DataFrame(train_vocab)
    train_vocab.to_csv(config['transform_data']['train_vocab_dir'],index=False)
    test_vocab = pd.DataFrame(test_vocab)
    test_vocab.to_csv(config['transform_data']['test_vocab_dir'],index=False)
    for i,vocab in enumerate(vocabs):
        embedded_sentences = []
        for sentence in vocab:
            embedded_sentence = [ft_model.get_word_vector(word) for word in sentence]

            if embedded_sentence:
                embedded_sentences.append(np.mean(embedded_sentence,axis=0))
            else:
                embedded_sentences.append(np.zeros(300))

        if i == 0:
            file_name = config['transform_data']['train_embedds_dir']
            df = pd.DataFrame(embedded_sentences)
            df = pd.concat([df,pd.Series(y_train)],axis=1)
            
        else:
            file_name = config['transform_data']['test_embedds_dir']
            df = pd.DataFrame(embedded_sentences)
            df = pd.concat([df,pd.Series(y_test)],axis=1)

        df.to_csv(file_name,index=False)
        
if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    transform_data(config_path=args.config)
