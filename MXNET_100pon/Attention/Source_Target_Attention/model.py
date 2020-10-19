import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from mxnet.gluon import rnn


class Encoder(gluon.Block):
    def __init__(self, vocab_size, embedding_dim, hidden_dim,
                 num_layers=1, batch_size=100, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        with self.name_scope():
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.batch_size = batch_size

            self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
            self.gru = rnn.GRU(
                hidden_dim, input_size=embedding_dim, layout="NTC")

    def forward(self, sequence):
        embedding = self.word_embeddings(sequence)
        # state.size() = (num_layers, batch_size, num_hidden)
        begin_state = mx.nd.random.uniform(
            shape=(self.num_layers, self.batch_size, self.hidden_dim))
        hs, h = self.gru(embedding, begin_state)
        return hs, h


class AttentionDecoder(gluon.Block):
    def __init__(self, vocab_size, embedding_dim,
                 hidden_dim, batch_size, **kwargs):
        super(AttentionDecoder, self).__init__(**kwargs)
        with self.name_scope():
            self.hidden_dim = hidden_dim
            self.batch_size = batch_size

            self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
            self.gru = rnn.GRU(
                hidden_dim, input_size=embedding_dim, layout="NTC")
            # 各系列のGRUの隠れ層とAttention層で計算したコンテ
            # hidden_dim*2としているのはキストベクトルを
            # つなぎ合わせることで長さが２倍になるため
            # self.hidden2linear = nn.Dense(vocab_size,in_units=hidden_dim * 2)
            self.hidden2linear = nn.Dense(
                vocab_size, in_units=hidden_dim * 2, flatten=False)

    def forward(self, sequence, hs, h):
        embedding = self.word_embeddings(sequence)
        output, state = self.gru(embedding, h)

        # Attention層
        # hs.size() = ([batch_size, 29, hidden_dim])
        # output.size() = ([batch_size, 10, hidden_dim])

        # bmmを使ってEncoder側の出力(hs)とDecoder側の出力(output)を
        # batchごとまとめて行列計算するために、Decoder側のoutputを
        # batchを固定して転置行列を取る
        # t_output.size() = ([batch_size, hidden_dim, 10])

        t_output = mx.ndarray.transpose(output, axes=(0, 2, 1))
        # s.size() = ([batch_size, 29, 10])
        s = mx.ndarray.linalg.gemm2(hs, t_output)

        # 列方向で和
        # attention_weight.size() = ([100, 29, 10])
        attention_weight = mx.nd.softmax(s, axis=1)

        # コンテキストベクトルをまとめるために入れ物を用意
        # c.size() = ([batch_size, 1, hidden_dim])

        c = mx.ndarray.zeros((self.batch_size, 1, self.hidden_dim))

        # 各層（Decoder側のGRU層は生成文字列が10文字なので10個ある）
        # におけるattention weightを取り出してforループ内でコンテキスト
        # ベクトルを１つずつ作成する
        # バッチ方向はまとめて計算できたのでバッチはそのまま

        for i in range(attention_weight.shape[2]):  # 10回ループ

            # attention_weight[:,:,i].size() = ([batch_size, 29])
            # i番目のGRU層に対するattention weightを取り出すが、テンソルの
            # サイズをhsと揃えるためにunsqueezeする
            # unsq_weight.size() = ([batch_size, 29, 1])
            unsq_weight = attention_weight[:, :, i].\
                reshape(attention_weight.shape[0],
                        attention_weight.shape[1], 1)

            # hsの各ベクトルをattention weightで重み付けする
            # weighted_hs.size() = ([batch_size, 29, hidden_num])
            weighted_hs = hs * unsq_weight

            # attention weightで重み付けされた各hsのベクトルをすべて足し合わせて
            # コンテキストベクトルを作成
            # weight_sum.size() = ([batch_size, 1, hidden_num])
            weight_sum = mx.nd.sum(weighted_hs, axis=1)\
                .reshape(self.batch_size,
                         1, self.hidden_dim)

            # c.size() = ([batch_size, i, hidden_num])
            c = mx.ndarray.concat(c, weight_sum, dim=1)

        # 箱として用意したzero要素が残っているのでスライスして削除
        # output.size() = ([batch_size, 10, hidden_dim*2])

        c = c[:, 1:, :]
        output = mx.nd.concat(output, c, dim=2)
        output = self.hidden2linear(output)

        return output, state, attention_weight
