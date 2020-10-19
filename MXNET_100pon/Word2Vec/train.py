from __future__ import print_function
import mxnet as mx
from mxnet import gluon

from mxnet import autograd as ag
from model import Word2Vec


def train_word2vec(net: Word2Vec, train_data: mx.io.io.NDArrayIter, loss_func,
                   epoch: int, ctx, trainer) -> (Word2Vec, list):
    general_losses = []

    for i in range(epoch):

        # Reset the train data iterator.
        train_data.reset()
        # Loop over the train data iterator.
        general_loss = 0

        for batch in train_data:
            # Splits train data into multiple slices along batch_axis
            # and copy each slice into a context.
            train_x = gluon.utils.split_and_load(
                batch.data[0], ctx_list=ctx, batch_axis=0)
            # Splits train labels into multiple slices along batch_axis
            # and copy each slice into a context.
            train_y = gluon.utils.split_and_load(
                batch.label[0], ctx_list=ctx, batch_axis=0)
            losses = []
            with ag.record():
                for x, y in zip(train_x, train_y):
                    zs = net(x)
                    # Computes softmax cross entropy loss.
                    for z in zs:
                        losses.append(loss_func(z, y))
                    loss = sum(losses)
                    loss.backward()
            # Make one step of parameter update. Trainer needs to know the
            # batch size of data to normalize the gradient by 1/batch_size.
            trainer.step(batch.data[0].shape[0])

            general_loss += sum(loss).asnumpy()[0]

        general_losses.append(general_loss)
        if i % 10 == 0:
            print("epoch_{i} loss: {loss}".format(i=i, loss=general_loss))

    return net, general_losses
