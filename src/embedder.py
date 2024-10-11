import fasttext
import fasttext.util
import pandas as pd
import sys
import numpy as np


def embedd_sentences(vocab_path: str, model_path: str) -> pd.DataFrame:
    vocab = pd.read_csv(vocab_path)
    ft_model = fasttext.load_model(model_path)
    embedded_sentences = []

    for sentence in vocab:
        embedded_sentence = [ft_model.get_word_vector(word) for word in sentence]

        if embedded_sentence:
            embedded_sentences.append(np.mean(embedded_sentence, axis=0))
        else:
            embedded_sentences.append(np.zeros(300))

    return pd.DataFrame(embedded_sentences)


if __name__ == '__main__':
    vocab_path = sys.argv[1]
    target_path = sys.argv[2]
    model_path = sys.argv[3]

    sentences_df = embedd_sentences(vocab_path, model_path)
    sentences_df.to_csv(target_path, index=False)
