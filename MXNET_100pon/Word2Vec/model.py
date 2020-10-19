from __future__ import print_function

from mxnet import gluon
from mxnet.gluon import nn

import mxnet.ndarray as F


class Word2Vec(gluon.Block):
    def __init__(self, hidden_dim, vocabulary_size, C, **kwargs):
        super(Word2Vec, self).__init__(**kwargs)
        with self.name_scope():
            self.embed = nn.Dense(
                hidden_dim, in_units=vocabulary_size, prefix="embed")
            self.C = C
            self.outs = []
            for c in range(C * 2):
                child_block = nn.Dense(
                    vocabulary_size, in_units=hidden_dim,
                    prefix="dense_"+str(c))

                self.register_child(child_block)
                self.outs.append(child_block)

    def forward(self, x):
        embed = self.embed(x)

        xs = []
        for i in range(self.C*2):
            x = self.outs[i](embed)
            x = F.softmax(x)
            xs.append(x)

        return xs

    def get_vec(self, input):
        return self.embed(input)
