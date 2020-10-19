import mxnet as mx
from mxnet import gluon
from mxnet import autograd as ag
import models

import datetime


def train(encoder: models.Encoder, attn_decoder: models.AttentionDecoder,
          train_data:  mx.io.io.NDArrayIter, epoch: int, criterion,
          opt="adam", learning_rate=0.03, save=True,
          magnitude=2.24) -> (models.Encoder, models.AttentionDecoder):

    print(encoder.summary)
    print(attn_decoder.summary)

    # set the context on GPU is available otherwise CPU
    # initialize the parameters of encoder
    ctx = [mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()]
    encoder.initialize(mx.init.Xavier(magnitude=magnitude), ctx=ctx)
    encoder_trainer = gluon.Trainer(encoder.collect_params(), opt, {
                                    'learning_rate': learning_rate})

    # set the context on GPU is available otherwise CPU
    # initialize the parameters of attention_decoder
    ctx = [mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()]
    attn_decoder.initialize(mx.init.Xavier(magnitude=magnitude), ctx=ctx)
    attn_decoder_trainer = gluon.Trainer(attn_decoder.collect_params(), opt, {
                                         'learning_rate': learning_rate})

    loss_log = []
    date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    for i in range(epoch):
        train_data.reset()
        for batch in train_data:
            data = gluon.utils.split_and_load(
                batch.data[0], ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(
                batch.label[0], ctx_list=ctx, batch_axis=0)

            with ag.record():
                for x, y in zip(data, label):
                    hs, h = encoder(x)

                    # Attention Decoderのインプット
                    source = y[:, :-1]
                    # Attention Decoderの正解データ
                    target = y[:, 1:]

                    decoder_output, _, attention_weight = attn_decoder(
                        source, hs, h)
                    # output = attn_decoder(source, hs, h)

                    loss = []
                    for j in range(decoder_output.shape[1]):
                        loss.append(
                            criterion(decoder_output[:, j, :], target[:, j]))

                    loss_sum = sum(loss)
                    loss_sum.backward()

            encoder_trainer.step(batch.data[0].shape[0])
            attn_decoder_trainer.step(batch.data[0].shape[0])

        # save and print loss
        loss_sum_sum = sum(loss_sum).asnumpy()[0]
        loss_log.append(loss_sum_sum)
        if i % 10 == 0:
            print("epoch_{i} loss: {loss}".format(i=i, loss=loss_sum_sum))

        # save the parameters
        if save:

            encoder.save_parameters(
                "encoder_{epoch}_{date}.params".format(epoch=epoch, date=date))
            attn_decoder.save_parameters(
                "attn_decoder_{epoch}_{date}.params".
                format(epoch=epoch, date=date))

    return encoder, attn_decoder
