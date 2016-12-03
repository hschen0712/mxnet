"""
@author: hschen0712
@description: bucket io for translate dataset
@time: 2016/11/22 16:21
"""

from trans_utils import read_tokens, build_vocab, text2id
from sklearn.cluster import KMeans
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


def gen_buckets(src_sents, trg_sents, num_buckets, the_vocab):
    lens = [(len(src_words), len(trg_words)) for src_words,trg_words in zip(src_sents, trg_sents)]

    lens = np.array(lens)
    kmeans = KMeans(n_clusters=num_buckets, random_state=1)
    assignment = kmeans.fit_predict(lens)
    buckets = []
    for i in range(num_buckets):
        idx = list(np.where(assignment==i)[0])
        sent_lens = [lens[i] for i in idx]
        buckets.append(zip(sent_lens, idx))
    return buckets


class ParallelCorpusIter(mx.io.DataIter):
    """
    Parallel Corpus Iter implemented with bucketing
    """
    def __init__(self, src_data_path, trg_data_path, src_vocab, trg_vocab, num_buckets, batch_size
                 ):
        super(ParallelCorpusIter, self).__init__()

        src_data, _ = read_tokens(src_data_path)
        trg_data, _ = read_tokens(trg_data_path)
        # use src data for building buckets
        src_data = [text2id(sent, src_vocab) for sent in src_data]
        trg_data = [text2id(sent, trg_vocab) for sent in trg_data]
        buckets = gen_buckets(src_data, trg_data, num_buckets, batch_size)

        # sort by sentence len in ascending order
        buckets.sort()
        # buckets stores the sentence lengths for each bucket
        self.buckets = buckets
        # data of buckets, e.g. self.data[0] may be the bucket(type list) of all sentences of length 1
        self.src_data = [[] for _ in buckets]
        self.trg_data = [[] for _ in buckets]
        # pre-allocate with the largest bucket for better memory sharing
        # max(buckets) is the max len of all sentences
        self.default_bucket_key = max(buckets)

        for src_sent, trg_sent in zip(src_data, trg_data):
            if len(src_sent) == 0:
                continue
            # look for a bucket that can hold this sentence exactly
            # bkt stands for sentence len of that bucket
            for i, bkt in enumerate(buckets):
                # bkt: bucket_key(or you can think of it as max sentence length of this bucket)
                # not all sentences in the same bucket are of the same length, but all less or equal to bkt
                if bkt >= len(src_sent):
                    self.src_data[i].append(src_sent)
                    self.trg_data[i].append(trg_sent)
                    break
                    # we just ignore the sentence it is longer than the maximum
                    # bucket size here
        # max length of target sentences in each bucket
        trg_buckets = []
        for bkt_sents in self.trg_data:
            lens = [len(sent) for sent in bkt_sents]
            max_bkt_len = max(lens)
            trg_buckets.append(max_bkt_len)
        self.trg_buckets = trg_buckets
        # self.vocab_size = len(vocab)
        # self.data_name = data_name
        # self.label_name = label_name
        #
        # convert data into ndarrays for better speed during training
        # note that not all sentences within a bucket are of equal len
        # shorter sentences are padded with 0's
        # x: sentences in this bucket, len(x): number of sentences in this bucket
        data = [np.zeros((len(x), buckets[i])) for i, x in enumerate(self.src_data)]
        # for each bucket
        for i_bucket in range(len(self.buckets)):
            # for each sentence in the same bucket
            for j in range(len(self.src_data[i_bucket])):
                sentence = self.src_data[i_bucket][j]
                data[i_bucket][j, :len(sentence)] = sentence
        # n_bucket x n_sentence x bucket_size
        self.src_data = data

        data = [np.zeros((len(x), trg_buckets[i])) for i, x in enumerate(self.trg_data)]
        for i_bucket in range(len(trg_buckets)):
            # for each sentence in the same bucket
            for j in range(len(self.trg_data[i_bucket])):
                sentence = self.trg_data[i_bucket][j]
                data[i_bucket][j, :len(sentence)] = sentence
                # n_bucket x n_sentence x bucket_size
        self.trg_data = data

        # Get the size of each bucket, so that we could sample
        # uniformly from the bucket
        # x: some bucket
        src_bucket_sizes = [len(x) for x in self.src_data]

        print("Summary of src dataset ==================")
        for bkt, size in zip(buckets, src_bucket_sizes):
            print("bucket of len %3d : %d samples" % (bkt, size))

        trg_bucket_sizes = [len(x) for x in self.trg_data]
        print("Summary of trg dataset ==================")
        for bkt, size in zip(buckets, trg_bucket_sizes):
            print("bucket of len %3d : %d samples" % (bkt, size))


        self.batch_size = batch_size

        self.make_data_iter_plan()

        # self.init_states = init_states
        # # x: output info, x[0]: name of output layer, x[1]: tuple, shape of output batch x num_hidden
        # self.init_state_arrays = [mx.nd.zeros(x[1]) for x in init_states]
        #
        # specify the shapes of input data and hidden layers
        self.provide_data = [('data', (batch_size, self.default_bucket_key))]
        # specify the shape of label
        self.provide_label = [('softmax_label', (self.batch_size, self.default_bucket_key))]

    def make_data_iter_plan(self):
        "make a random data iteration plan"
        # truncate each bucket into multiple of batch-size
        bucket_n_batches = []
        # for each bucket
        for i in range(len(self.src_data)):
            # count how many batches there are in each bucket
            bucket_n_batches.append(len(self.src_data[i]) / self.batch_size)
            # throw away remaining samples that are not enough for a batch
            self.src_data[i] = self.src_data[i][:int(bucket_n_batches[i] * self.batch_size)]
            self.trg_data[i] = self.trg_data[i][:int(bucket_n_batches[i] * self.batch_size)]

        # concatenate in horizontal direction
        # bucket_plan maps each batch to its correspoding bucket index
        # suppose that batches are [1,3,2], then bucket_plan is [0,1,1,1,2,2]
        bucket_plan = np.hstack([np.zeros(n, int) + i for i, n in enumerate(bucket_n_batches)])
        # shuffle buckets, and then shuffle seqs in each bucket
        np.random.shuffle(bucket_plan)

        bucket_idx_all = [np.random.permutation(len(x)) for x in self.src_data]

        self.bucket_plan = bucket_plan
        self.bucket_idx_all = bucket_idx_all
        # index for current batch in each bucket
        self.bucket_curr_idx = [0 for x in self.src_data]

        # buffers for batch data in each bucket
        self.data_buffer = []
        self.label_buffer = []
        for i_bucket in range(len(self.src_data)):
            # batch_size x bucket_size of i'th bucket
            data = np.zeros((self.batch_size, self.buckets[i_bucket]))
            label = np.zeros((self.batch_size, self.trg_buckets[i_bucket]))
            self.data_buffer.append(data)
            self.label_buffer.append(label)

    def __iter__(self):

        for i_bucket in self.bucket_plan:
            data = self.data_buffer[i_bucket]
            i_idx = self.bucket_curr_idx[i_bucket]
            # index range for shuffled data in a batch of sentences
            idx = self.bucket_idx_all[i_bucket][i_idx:i_idx + self.batch_size]
            self.bucket_curr_idx[i_bucket] += self.batch_size


            # init_state_names = [x[0] for x in self.init_states]
            data[:] = self.src_data[i_bucket][idx]

            for sentence in data:
                assert len(sentence) == self.buckets[i_bucket]

            label = self.label_buffer[i_bucket]

            label[:] = self.trg_data[i_bucket][idx]

            # data_all = [mx.nd.array(data)] + self.init_state_arrays
            data_all = [mx.nd.array(data)]
            label_all = [mx.nd.array(label)]
            # data_names = ['data'] + init_state_names
            data_names = ['data']
            label_names = ['softmax_label']

            data_batch = SimpleBatch(data_names, data_all, label_names, label_all,
                                     self.buckets[i_bucket])
            yield data_batch

if __name__ == "__main__":
    _, word_lst = read_tokens("data/english1w.txt")
    build_vocab(word_lst, 20000, "data/eng.vcb.pkl")
    _, word_lst = read_tokens("data/chinese1w.txt")
    build_vocab(word_lst, 20000, "data/chn.vcb.pkl")
    src_vocab = pickle.load(open("data/chn.vcb.pkl", "r"))
    trg_vocab = pickle.load(open("data/eng.vcb.pkl", "r"))
    batch_size = 32
    num_buckets = 6
    data_train = ParallelCorpusIter("data/chinese8k.txt", "data/english8k.txt", src_vocab, trg_vocab, num_buckets, batch_size)
    for batch in data_train:
        print "data:",batch.data[0].shape
        print "label:",batch.label[0].shape
