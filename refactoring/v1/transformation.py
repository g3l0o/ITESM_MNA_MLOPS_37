import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
import fasttext
import fasttext.util
from gensim.models import FastText
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os

def obtain_vocabs(df,test_size,random_state):
    random_state = 2
    X = df['words'].values
    y = df['Category ID'].values

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_size,random_state=random_state)


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

    return train_vocab,test_vocab,np.array(y_train),np.array(y_test)

def embedd_sentences(vocab):
    
    current_dir = os.path.dirname(__file__)
    
    model_path = os.path.join(current_dir, '../v1/cc.en.300.bin')
    
    ft_model = fasttext.load_model(model_path)
    embedded_sentences = []

    for sentence in vocab:
        embedded_sentence = [ft_model.get_word_vector(word) for word in sentence]

        if embedded_sentence:
            embedded_sentences.append(np.mean(embedded_sentence,axis=0))
        else:
            embedded_sentences.append(np.zeros(300))

    return np.array(embedded_sentences)


def wordClouds(x,y):
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
    fig,axs = plt.subplots(rows,cols,figsize=(12,6))

    axs = axs.flat if categories > 1 else [axs]


    # Crear una nube de palabras para cada categoría
    for key,val in words.items():
        wordcloud = WordCloud().generate(val)
        ax = axs[key]
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.set_title(labels[key])
        ax.axis('off')  # Ocultar los ejes
    
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0, hspace=0.4)
    plt.tight_layout()
    plt.show()