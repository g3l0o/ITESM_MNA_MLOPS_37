import sys

import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np


def obtain_vocabs(df_path:str, test_size=0.3, random_state = 2):

    df = pd.read_csv(df_path)

    X = df['words'].values
    y = df['category_id'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    diccionario = Counter()
    for i in range(len(X_train)):
        diccionario.update(X_train[i])

    min_freq = 2
    mi_dicc = {}
    for key, val in diccionario.items():
        if val > min_freq:
            mi_dicc[key] = val

    train_vocab = []
    for sentence in X_train:
        train_vocab.append([word for word in sentence if word in mi_dicc])

    test_vocab = []
    for sentence in X_test:
        test_vocab.append([word for word in sentence if word in mi_dicc])

    return train_vocab, test_vocab, np.array(y_train), np.array(y_test)


if __name__ == '__main__':
    data_path = sys.argv[1]
    output_train_features = sys.argv[2]
    output_test_features = sys.argv[3]
    output_train_target = sys.argv[4]
    output_test_target = sys.argv[5]

    X_train, X_test, y_train, y_test = obtain_vocabs(data_path)
    pd.DataFrame(X_train).to_csv(output_train_features, index=False)
    pd.DataFrame(X_test).to_csv(output_test_features, index=False)
    pd.DataFrame(y_train).to_csv(output_train_target, index=False)
    pd.DataFrame(y_test).to_csv(output_test_target, index=False)