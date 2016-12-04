"""
@author: hschen0712
@description: bucket io for translate dataset
@time: 2016/11/22 16:21
"""

from trans_utils import read_tokens, build_vocab, text2id
import numpy as np
import mxnet as mx
import cPickle as pickle

# The interface of a data iter that works for bucketing
#
# DataIter
#   - default_bucket_key: the bucket key for the default symbol.
#
# DataBatch
#   - provide_data: same as DataIter, but specific to this batch
#   - provide_label: same as DataIter, but specific to this batch
#   - bucket_key: the key for the bucket that should be used for this batch

class SimpleBatch(object):
    def __init__(self, data_names, data, label_names, label, bucket_key):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names
        self.bucket_key = bucket_key

        self.pad = 0
        self.index = None  # TODO: what is index?

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]


class DummyIter(mx.io.DataIter):
    "A dummy iterator that always return the same batch, used for speed testing"

    def __init__(self, real_iter):
        super(DummyIter, self).__init__()
        self.real_iter = real_iter
        self.provide_data = real_iter.provide_data
        self.provide_label = real_iter.provide_label
        self.batch_size = real_iter.batch_size

        for batch in real_iter:
            self.the_batch = batch
            break

    def __iter__(self):
        return self

    def next(self):
        return self.the_batch


class ParallelCorpusIter(mx.io.DataIter):
    """
    Parallel Corpus Iter
    """
    def __init__(self, src_data_path, trg_data_path, src_vocab, trg_vocab,
                 enc_init_states, batch_size, bucket_key=(50, 40)):
        super(ParallelCorpusIter, self).__init__()
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.src_vocab_size = len(src_vocab)
        self.trg_vocab_size = len(trg_vocab)

        src_data, _ = read_tokens(src_data_path)
        trg_data, _ = read_tokens(trg_data_path)
        assert len(src_data) == len(trg_data)
        # use src data for building buckets
        self.src_data = [text2id(sent, src_vocab) for sent in src_data]
        self.trg_data = [text2id(sent, trg_vocab) for sent in trg_data]

        # pre-allocate with the largest bucket for better memory sharing
        self.default_bucket_key = bucket_key
        enc_len, dec_len = self.default_bucket_key

        # convert data into ndarrays for better speed during training
        # note that not all sentences within a bucket are of equal len
        # longer sentences are truncated to bucket_size
        # and shorter sentences are padded with 0's
        src_data = np.zeros((len(src_data), self.default_bucket_key[0]))
        trg_data = np.zeros((len(trg_data), self.default_bucket_key[1]))
        src_mask = np.zeros((len(src_data), self.default_bucket_key[0]))
        trg_mask = np.zeros((len(trg_data), self.default_bucket_key[1]))

        # for each sentence in the bucket
        for idx, (src_sent, trg_sent) in enumerate(zip(self.src_data, self.trg_data)):
            if len(src_sent) > enc_len:
                src_sent = src_sent[:enc_len]
            src_sent = np.array(src_sent)
            src_data[idx, :len(src_sent)] = src_sent
            src_mask[idx, :len(src_sent)] = np.asarray(src_sent>0, int)
            if len(trg_sent) > dec_len:
                trg_sent = trg_sent[:dec_len]
            trg_sent = np.array(trg_sent)
            trg_data[idx, :len(trg_sent)] = trg_sent
            trg_mask[idx, :len(trg_sent)] = np.asarray(trg_sent > 0, int)
        # n_sentence x enc_len
        self.src_data = src_data
        # n_sentence x dec_len
        self.trg_data = trg_data
        self.src_mask = src_mask
        self.trg_mask = trg_mask

        self.batch_size = batch_size

        self.enc_init_states = enc_init_states
        # self.dec_init_states = dec_init_states
        # x: output info, x[0]: name of output layer, x[1]: tuple, shape of output
        self.enc_init_state_names = [x[0] for x in enc_init_states]
        self.enc_init_state_arrays = [mx.nd.zeros(x[1]) for x in enc_init_states]
        # self.dec_init_state_names = [x[0] for x in dec_init_states]
        # self.dec_init_state_arrays = [mx.nd.zeros(x[1]) for x in dec_init_states]

        self.make_iter_plan()
        # specify the shapes of input data and hidden layers
        self.provide_data = [('src_data', (batch_size, self.default_bucket_key[0])),
                             ('src_mask', (batch_size, self.default_bucket_key[0])),
                             ('trg_mask', (batch_size, self.default_bucket_key[1]))] + enc_init_states
        # specify the shape of label
        self.provide_label = [('trg_data', (batch_size, self.default_bucket_key[1]))]

    def make_iter_plan(self):
        indices = range(len(self.src_data))
        np.random.shuffle(indices)
        self.batch_indices = [indices[idx*self.batch_size:(idx+1)*self.batch_size] for idx in range(len(indices) // self.batch_size)]


    def __iter__(self):
        for batch_idx in self.batch_indices:
            src_data = self.src_data[batch_idx]
            src_mask = self.src_mask[batch_idx]
            trg_data = self.trg_data[batch_idx]
            trg_mask = self.trg_mask[batch_idx]

            data_all = [mx.nd.array(src_data), mx.nd.array(src_mask), mx.nd.array(trg_mask)] + self.enc_init_state_arrays
            label_all = [mx.nd.array(trg_data)]
            data_names = ['src_data', 'src_mask', 'trg_mask'] + self.enc_init_state_names
            label_names = ['trg_data']

            data_batch = SimpleBatch(data_names, data_all, label_names, label_all,
                                     self.default_bucket_key)
            yield data_batch

if __name__ == "__main__":
    _, word_lst = read_tokens("data/english1w.txt")
    build_vocab(word_lst, 20000, "data/eng.vcb.pkl")
    _, word_lst = read_tokens("data/chinese1w.txt")
    build_vocab(word_lst, 20000, "data/chn.vcb.pkl")
    src_vocab = pickle.load(open("data/chn.vcb.pkl", "r"))
    trg_vocab = pickle.load(open("data/eng.vcb.pkl", "r"))
    batch_size = 32
    num_lstm_layer = 1
    num_hidden = 100
    init_h = [('l%d_init_h' % l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    init_states = init_h
    data_train = ParallelCorpusIter("data/chinese8k.txt", "data/english8k.txt", src_vocab, trg_vocab, init_states, batch_size)
    print "default bucket key:",data_train.default_bucket_key
    count = 1
    # print the whole array
    # np.set_printoptions(threshold='nan')
    for batch_ in data_train:
        for data, name in zip(batch_.data, batch_.data_names):
            print "name:%s, shape:%s" % \
                  (name, str(data.shape))
            # print "%s:"%name, data.asnumpy()
        for label, name in zip(batch_.label, batch_.label_names):
            print "name:%s, shape:%s" % \
                  (name, str(label.shape))
            # print "%s:" % name, label.asnumpy()
        count += 1
    print "total %d batches" % count

