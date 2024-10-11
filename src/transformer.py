import sys

import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
import fasttext
import fasttext.util
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os
from datetime import datetime


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


def embedd_sentences(vocab, dataset):
    current_dir = os.path.dirname(__file__)

    model_path = os.path.join(current_dir, '../v1/cc.en.300.bin')

    ft_model = fasttext.load_model(model_path)
    embedded_sentences = []

    for sentence in vocab:
        embedded_sentence = [ft_model.get_word_vector(word) for word in sentence]

        if embedded_sentence:
            embedded_sentences.append(np.mean(embedded_sentence, axis=0))
        else:
            embedded_sentences.append(np.zeros(300))

    time = ''.join(str(datetime.now())[:-7])
    current_dir = os.path.dirname(__file__)
    data_path = os.path.join(current_dir, '../../data/final/')
    df = pd.DataFrame(embedded_sentences)
    df.to_csv(f'{data_path + time}_{dataset}.csv', index=False)
    return np.array(embedded_sentences)


def wordClouds(x, y):
    words = {}
    labels = {0: 'Mobile Phones',
              1: 'TVs',
              2: 'CPUs',
              3: 'Digital Cameras',
              4: 'Microwaves',
              5: 'Dishwashers',
              6: 'Washing Machines',
              7: 'Freezers',
              8: 'Fridge Freezers',
              9: 'Fridges'}

    for i in range(len(y)):
        text = ' '.join(x[i])
        if y[i] in words.keys():
            words[y[i]] += text + ' '
        else:
            words[y[i]] = text

    rows = 2
    categories = len(words)
    cols = (categories + 1) // 2
    fig, axs = plt.subplots(rows, cols, figsize=(12, 6))

    axs = axs.flat if categories > 1 else [axs]

    # Crear una nube de palabras para cada categoría
    for key, val in words.items():
        wordcloud = WordCloud().generate(val)
        ax = axs[key]
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.set_title(labels[key])
        ax.axis('off')  # Ocultar los ejes

    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0, hspace=0.4)
    plt.tight_layout()
    plt.show()


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