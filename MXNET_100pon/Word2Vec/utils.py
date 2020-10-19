from __future__ import print_function
import numpy as np


def tokenize_corpus(corpus):
    tokens = [x.split() for x in corpus]
    return tokens


def get_idx_pairs(window_size: int, word2idx: {int: str},
                  tokenized_corpus: [[str]]) -> [[int]]:

    idx_pairs = []
    # for each sentence
    for sentence in tokenized_corpus:
        indices = [word2idx[word] for word in sentence]
        # for each word, threated as center word
        for center_word_pos in range(len(indices)):
            # for each window position
            for w in range(-window_size, window_size + 1):
                context_word_pos = center_word_pos + w
                # make soure not jump out sentence
                if context_word_pos < 0 or context_word_pos >= len(indices) \
                   or center_word_pos == context_word_pos:
                    continue
                context_word_idx = indices[context_word_pos]
                idx_pairs.append((indices[center_word_pos], context_word_idx))

    # it will be useful to have this as numpy array
    idx_pairs = np.array(idx_pairs)

    return idx_pairs


def get_input_layer(word_idx, vocabulary_size):
    x = np.zeros(vocabulary_size)
    x[word_idx] = 1.0
    return x


def data_load(idx_pairs: [[int]], vocabulary_size: int) \
        -> (np.ndarray, np.ndarray):

    data = []
    target = []
    for d, t in idx_pairs:
        data.append(get_input_layer(d, vocabulary_size))
        target.append(get_input_layer(t, vocabulary_size))

    data = np.array(data)
    target = np.array(target)

    return data, target


def file2corpus(fpath: str):
    corpus = []
    with open(fpath, "r") as f:
        for l in f.readlines():
            corpus.append(l.rstrip('\n'))

    return corpus
