"""
@author: hschen0712
@description: modified run_cell_demo.py, added language model training with GRU
@time: 2016/11/22 15:15
"""

import os
import numpy as np
import mxnet as mx
from mxnet.metric import EvalMetric

from bucket_io_kmeans import BucketSentenceIter, default_build_vocab


data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))

# this function only calculates the ppl of last batch, while our goal is to
# calculate the average ppl of all batches
def Perplexity(label, pred):
    # TODO(tofix): we make a transpose of label here, because when
    # using the RNN cell, we called swap axis to the data.
    # label:  bs x n_step
    # label.T.reshape((-1,)) means transpose and flatten
    label = label.T.reshape((-1,))
    loss = 0.
    # pred.shape: (n_step * bs) x vocab_size
    for i in range(pred.shape[0]):
        # skip padding 0's
        if int(label[i]) != 0:
            loss += -np.log(max(1e-10, pred[i][int(label[i])]))
    return np.exp(loss / label.size)


if __name__ == '__main__':
    batch_size = 32
    num_hidden = 200
    num_embed = 200
    num_lstm_layer = 2
    num_buckets = 6

    num_epoch = 5
    learning_rate = 0.01
    momentum = 0.0

    contexts = mx.context.gpu(0)
    vocab = default_build_vocab(os.path.join(data_dir, 'ptb.train.txt'))

    init_h = [('GRU_init_h', (batch_size, num_lstm_layer, num_hidden))]

    data_train = BucketSentenceIter(os.path.join(data_dir, 'ptb.train.txt'),
                                    vocab, num_buckets, batch_size, init_h)
    data_val = BucketSentenceIter(os.path.join(data_dir, 'ptb.valid.txt'),
                                  vocab, num_buckets, batch_size, init_h)

    def sym_gen(seq_len):
        # data, label, and sequence_length are provided in batch iterator
        data = mx.sym.Variable('data')
        label = mx.sym.Variable('softmax_label')
        # shape: batch_size x seq_len x num_embed
        embed = mx.sym.Embedding(data=data, input_dim=len(vocab),
                                 output_dim=num_embed, name='embed')
        # TODO(tofix)
        # The inputs and labels from IO are all in batch-major.
        # Perform dimshuffle to convert them into time-major
        embed_tm = mx.sym.SwapAxis(embed, dim1=0, dim2=1)
        label_tm = mx.sym.SwapAxis(label, dim1=0, dim2=1)
        sequence_length = mx.sym.Variable('sequence_length')
        masked_embed = mx.sym.SequenceMask(data=embed_tm, sequence_length=sequence_length,
                                           use_sequence_length=True, name="mask")


        # TODO(tofix)
        # Create transformed RNN initial states. Normally we do
        # no need to do this. But the RNN symbol expects the state
        # to be time-major shape layout, while the current mxnet
        # IO and high-level training logic assume everything from
        # the data iter have batch_size as the first dimension.
        # So until we have extended our IO and training logic to
        # support this more general case, this dummy axis swap is
        # needed.
        rnn_h_init = mx.sym.SwapAxis(mx.sym.Variable('GRU_init_h'),
                                     dim1=0, dim2=1)

        # TODO(tofix)
        # currently all the LSTM parameters are concatenated as
        # a huge vector, and named '<name>_parameters'. By default
        # mxnet initializer does not know how to initilize this
        # guy because its name does not ends with _weight or _bias
        # or anything familiar. Here we just use a temp workaround
        # to create a variable and name it as LSTM_bias to get
        # this demo running. Note by default bias is initialized
        # as zeros, so this is not a good scheme. But calling it
        # LSTM_weight is not good, as this is 1D vector, while
        # the initialization scheme of a weight parameter needs
        # at least two dimensions.
        rnn_params = mx.sym.Variable('GRU_bias')

        # RNN cell takes input of shape (time, batch, feature)
        # In GRU mode,we don;t need to provide argument state_cell
        rnn = mx.sym.RNN(data=masked_embed, state_size=num_hidden,
                         num_layers=num_lstm_layer, mode='gru',
                         name='gru',
                         # The following params can be omitted
                         # provided we do not need to apply the
                         # workarounds mentioned above
                         state=rnn_h_init,
                         parameters=rnn_params)

        # the RNN cell output is of shape (time, batch, dim)
        # if we need the states and cell states in the last time
        # step (e.g. when building encoder-decoder models), we
        # can set state_outputs=True, and the RNN cell will have
        # extra outputs: rnn['LSTM_output'], rnn['LSTM_state']
        # and for LSTM, also rnn['LSTM_state_cell']

        # now we collapse the time and batch dimension to do the
        # final linear logistic regression prediction
        hidden = mx.sym.Reshape(data=rnn, shape=(-1, num_hidden))
        label_cl = mx.sym.Reshape(data=label_tm, shape=(-1,))

        pred = mx.sym.FullyConnected(data=hidden, num_hidden=len(vocab),
                                     name='pred')
        sm = mx.sym.SoftmaxOutput(data=pred, label=label_cl, name='softmax')

        return sm

    model = mx.model.FeedForward(ctx=contexts,
                                 symbol=sym_gen,
                                 num_epoch=num_epoch,
                                 learning_rate=learning_rate,
                                 momentum=momentum,
                                 wd=0.00001,
                                 initializer=mx.init.Xavier(factor_type="in", magnitude=2.34))

    # mod = mx.mod.BucketingModule(, default_bucket_key=data_train.default_bucket_key,
    #                                  context=contexts)

    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)

    model.fit(X=data_train, eval_data=data_val,
              eval_metric=mx.metric.np(Perplexity),
              batch_end_callback=mx.callback.Speedometer(batch_size, 50), )
