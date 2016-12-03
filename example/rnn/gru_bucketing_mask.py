"""
@author: hschen0712
@description:
@time: 2016/12/03 12:00
"""

# pylint: disable=C0111,too-many-arguments,too-many-instance-attributes,too-many-locals,redefined-outer-name,fixme
# pylint: disable=superfluous-parens, no-member, invalid-name
import numpy as np
import mxnet as mx

from gru import gru_unroll
from bucket_io_mask import BucketSentenceIter, default_build_vocab

def Perplexity(label, pred):
    # transpose and flatten in row major
    # label:  bs x n_step
    label = label.T.reshape((-1,))
    loss = 0.
    # pred.shape: (n_step * bs) x vocab_size
    count = 0.
    for i in range(pred.shape[0]):
        if int(label[i]) != 0:
            loss += -np.log(max(1e-10, pred[i][int(label[i])]))
            count += 1.

    return np.exp(loss / count)

if __name__ == '__main__':
    batch_size = 32
    num_hidden = 200
    num_embed = 200
    num_lstm_layer = 1
    num_buckets = 6

    num_epoch = 25
    learning_rate = 0.01
    momentum = 0.0

    # dummy data is used to test speed without IO
    dummy_data = False

    contexts = [mx.context.gpu(i) for i in range(1)]
    # contexts = mx.context.cpu()

    vocab = default_build_vocab("./data/ptb.train.txt")

    def sym_gen(seq_len):
        return gru_unroll(num_lstm_layer, seq_len, len(vocab),
                           num_hidden=num_hidden, num_embed=num_embed,
                           num_label=len(vocab), use_mask=True)

    init_h = [('l%d_init_h'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]

    data_train = BucketSentenceIter("./data/ptb.train.txt", vocab,
                                    num_buckets, batch_size, init_h)
    data_val = BucketSentenceIter("./data/ptb.valid.txt", vocab,
                                  num_buckets, batch_size, init_h)

    if dummy_data:
        data_train = DummyIter(data_train)
        data_val = DummyIter(data_val)

    model = mx.model.FeedForward(ctx=contexts,
                                 symbol=sym_gen,
                                 num_epoch=num_epoch,
                                 learning_rate=learning_rate,
                                 momentum=momentum,
                                 wd=0.00001,
                                 initializer=mx.init.Xavier(factor_type="in", magnitude=2.34))

    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)

    model.fit(X=data_train, eval_data=data_val,
              eval_metric = mx.metric.np(Perplexity),
              batch_end_callback=mx.callback.Speedometer(batch_size, 50))

