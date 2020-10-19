from __future__ import print_function

import mxnet as mx
from mxnet import gluon

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns

from models import Word2Vec
from train import train_word2vec
from utils import file2corpus, tokenize_corpus, get_idx_pairs
from utils import get_input_layer, data_load


def main():

    hidden_dim = 20
    opt = "adam"  # SGD, Adam
    C = 3  # word2vec window size satisfying C >= 1
    # x_length = 1 + C * 2  # training label length
    batch_size = 3
    epoch = 150
    f_path = "sample.txt"

    visualize = True

    # --------------------------
    corpus = file2corpus(f_path)
    tokenized_corpus = tokenize_corpus(corpus)
    vocabulary = list(set(sum(tokenized_corpus, [])))

    word2idx = {w: idx for (idx, w) in enumerate(vocabulary)}
    idx2word = {idx: w for (idx, w) in enumerate(vocabulary)}

    vocabulary_size = len(vocabulary)

    idx_pairs = get_idx_pairs(C, word2idx, tokenized_corpus)

    # --------------------------

    net = Word2Vec(hidden_dim, vocabulary_size, C)
    print(net.summary)

    data, target = data_load(idx_pairs, vocabulary_size)
    train_data = mx.io.NDArrayIter(
        data, target, batch_size=batch_size, shuffle=False)

    # -----------------------
    # set the context on GPU is available otherwise CPU
    ctx = [mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()]
    net.initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
    trainer = gluon.Trainer(net.collect_params(), opt, {'learning_rate': 0.03})
    loss_func = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=False)

    # --------------------------
    net, loss_log = train_word2vec(
        net, train_data, loss_func, epoch, ctx, trainer)

    plt.plot(loss_log)
    plt.title("epoch: {epoch}, dim: {dim}".format(epoch=epoch, dim=hidden_dim))
    plt.savefig("word2vec_loss_epoch_{epoch}_dim_{dim}.png".format(
        epoch=epoch, dim=hidden_dim))
    # plt.show()

    # -----------------------------

    if visualize:
        word2score = {}
        for k, v in idx2word.items():
            word2score[v] = net.get_vec(mx.nd.array(get_input_layer(
                k, vocabulary_size).reshape(1, vocabulary_size))).asnumpy()

        z = [v[0] for v in word2score.values()]

        corrdf = pd.DataFrame(z, index=word2score.keys()).T.corr()
        plt.figure()
        plt.title("epoch: {epoch}, dim: {dim}".format(
            epoch=epoch, dim=hidden_dim))
        sns.heatmap(corrdf)
        plt.savefig("word2vec_corr_epoch_{epoch}_dim_{dim}.png".format(
            epoch=epoch, dim=hidden_dim))
        # plt.show()

        fig, ax = plt.subplots()
        c = ax.pcolor(z)
        fig.colorbar(c, ax=ax)
        ax.set_yticks(np.arange(vocabulary_size)+0.5, minor=False)
        ax.set_xticks(np.arange(hidden_dim)+0.5, minor=False)
        ax.set_xticklabels(list(range(hidden_dim)))
        ax.set_yticklabels(word2score.keys())
        plt.title("epoch: {epoch}, dim: {dim}".format(
            epoch=epoch, dim=hidden_dim))
        plt.savefig("word2vec_epoch_{epoch}_dim_{dim}.png".format(
            epoch=epoch, dim=hidden_dim))
        # plt.show()
