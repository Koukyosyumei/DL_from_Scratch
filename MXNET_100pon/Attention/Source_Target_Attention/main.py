import numpy as np
from sklearn.model_selection import train_test_split

import mxnet as mx
from mxnet import gluon

import model
import utils
from train import train


def main():
    # -------- hyper params --------------
    file_path = "nlp_sample.txt"
    embedding_dim = 200
    hidden_dim = 128
    BATCH_NUM = 100

    epoch = 10
    # 損失関数
    criterion = gluon.loss.SoftmaxCrossEntropyLoss()
    # optimize
    opt = "adam"

    save = True

    # ----- dataの用意 ---------

    input_date, output_date = utils.date_load(file_path)
    # inputとoutputの系列の長さを取得
    # すべて長さが同じなので、0番目の要素でlenを取ってます
    # paddingする必要は、このデータに対してはない
    # input_len = len(input_date[0])  # 29
    # output_len = len(output_date[0])  # 10

    input_data, output_data, char2id, id2char = utils.create_corpus(
        input_date, output_date)
    vocab_size = len(char2id)

    # 7:3でtrainとtestに分ける
    train_x, test_x, train_y, test_y = train_test_split(
        input_data, output_data, train_size=0.7)

    train_x = np.array(train_x)
    train_y = np.array(train_y)
    train_data = mx.io.NDArrayIter(train_x, train_y, BATCH_NUM, shuffle=False)

    # -------- training ---------------

    encoder = model.Encoder(vocab_size, embedding_dim, hidden_dim)
    attn_decoder = model.AttentionDecoder(
        vocab_size, embedding_dim, hidden_dim, BATCH_NUM)

    encoder, attn_decoder = train(encoder, attn_decoder, train_data,
                                  epoch, criterion, opt=opt, save=save)


if __name__ == "__main__":
    main()
