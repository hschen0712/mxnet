"""
@author: hschen0712
@description: create buckets using kmeans clustering
@time: 2016/11/23 11:47
"""

# import sys

# sys.path.insert(0, "../../python")
import numpy as np
import mxnet as mx
import math
from sklearn.cluster import KMeans


# The interface of a data iter that works for bucketing
#
# DataIter
#   - default_bucket_key: the bucket key for the default symbol.
#
# DataBatch
#   - provide_data: same as DataIter, but specific to this batch
#   - provide_label: same as DataIter, but specific to this batch
#   - bucket_key: the key for the bucket that should be used for this batch

def default_read_content(path):
    with open(path) as ins:
        content = ins.read()
        content = content.replace('\n', ' <eos> ').replace('. ', ' <eos> ')
        return content


def default_build_vocab(path):
    content = default_read_content(path)
    content = content.split(' ')
    idx = 1  # 0 is left for zero-padding
    the_vocab = {}
    the_vocab[' '] = 0  # put a dummy element here so that len(vocab) is correct
    for word in content:
        if len(word) == 0:
            continue
        if not word in the_vocab:
            the_vocab[word] = idx
            idx += 1
    return the_vocab


def default_text2id(sentence, the_vocab):
    words = sentence.split(' ')
    words = [the_vocab[w] for w in words if len(w) > 0]
    return words


def gen_buckets_kmeans(sentences, num_buckets, the_vocab):
    lens = []
    for sentence in sentences:
        words = default_text2id(sentence, the_vocab)
        lens.append(len(words))
    lens = np.array(lens)
    kmeans = KMeans(n_clusters=num_buckets, random_state=1)
    assignment = kmeans.fit_predict(lens[:, None])
    buckets = []
    for i in range(num_buckets):
        idx = list(np.where(assignment==i)[0])
        sent_lens = [lens[i] for i in idx]
        buckets.append(zip(sent_lens, idx))
    return buckets


class ModelParallelBatch(object):
    """Batch used for model parallelism"""

    def __init__(self, data, bucket_key):
        self.data = np.array(data)
        self.bucket_key = bucket_key


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


class BucketSentenceIter(mx.io.DataIter):
    def __init__(self, path, vocab, num_buckets, batch_size,
                 init_states, data_name='data', label_name='label',
                 seperate_char=' <eos> ', text2id=None, read_content=None, model_parallel=False):
        super(BucketSentenceIter, self).__init__()

        if text2id == None:
            self.text2id = default_text2id
        else:
            self.text2id = text2id
        if read_content == None:
            self.read_content = default_read_content
        else:
            self.read_content = read_content
        content = self.read_content(path)
        sentences = content.split(seperate_char)
        self.num_buckets = num_buckets
        # bucket: (sent_len, idx)
        buckets = gen_buckets_kmeans(sentences, num_buckets, vocab)
        # sequence lengths of sentences in each bucket
        self.sequence_length = []
        for bucket in buckets:
            self.sequence_length.append(np.array([sent_len for sent_len, _ in bucket]))
        # data of buckets, e.g. self.data[0] may be the bucket(type list) of all sentences of length 1
        # equivalent to: [[]] * len(buckets)
        self.data = [[] for _ in range(len(buckets))]

        self.vocab_size = len(vocab)
        self.data_name = data_name
        self.label_name = label_name
        self.model_parallel = model_parallel
        maxlen_per_bucket = []
        for i in range(len(buckets)):
            bucket = buckets[i]
            maxlen = -1
            for sent_len, idx in bucket:
                if sent_len > maxlen:
                    maxlen = sent_len
                sentence = sentences[idx]
                sentence = self.text2id(sentence, vocab)
                self.data[i].append(sentence)
            maxlen_per_bucket.append(maxlen)


        # buckets stores the sentence lengths for each bucket
        self.buckets = buckets

        # pre-allocate with the largest bucket for better memory sharing
        self.default_bucket_key = max(maxlen_per_bucket) + 1
        self.maxlen_per_bucket = maxlen_per_bucket
        # convert data into ndarrays for better speed during training
        # note that not all sentences within a bucket are of equal len
        # shorter sentences are padded with 0's
        # x: sentences in this bucket, len(x): number of sentences in this bucket
        data = [np.zeros((len(x), maxlen_per_bucket[i])) for i, x in enumerate(self.data)]
        # for each bucket
        for i_bucket in range(len(self.buckets)):
            # for each sentence in the same bucket
            for j in range(len(self.data[i_bucket])):
                sentence = self.data[i_bucket][j]
                data[i_bucket][j, :len(sentence)] = sentence
        # n_bucket x n_sentence x bucket_size
        self.data = data

        # Get the size of each bucket, so that we could sample
        # uniformly from the bucket
        # x: some bucket
        bucket_sizes = [len(x) for x in self.data]

        self.batch_size = batch_size
        self.make_data_iter_plan()

        print("Summary of dataset ==================")
        max_lens_bucket = []
        min_lens_bucket = []
        avg_lens_bucket = []
        for bucket in buckets:
            max_len = -1
            min_len = np.inf
            sum_ = 0.
            for sent_len, _ in bucket:
                if sent_len > max_len:
                    max_len = sent_len
                if sent_len < min_len:
                    min_len = sent_len
                sum_ += sent_len
            avg_len = sum_ / len(bucket)
            max_lens_bucket.append(max_len)
            min_lens_bucket.append(min_len)
            avg_lens_bucket.append(avg_len)

        self.avg_lens_bucket = [math.ceil(len_) for len_ in avg_lens_bucket]

        for i in range(len(buckets)):
            print("bucket %d : %d samples, max len:%d, min len:%d, avg len:%f"
                  % (i, bucket_sizes[i], max_lens_bucket[i],
                     min_lens_bucket[i], avg_lens_bucket[i]))
        print "batch size is:%d,total number of batches:%d"%(self.batch_size,self.num_batches)

        self.init_states = init_states
        # x: output info, x[0]: name of output layer, x[1]: tuple, shape of output batch x num_hidden
        self.init_state_arrays = [mx.nd.zeros(x[1]) for x in init_states]

        # specify the shapes of input data and hidden layers
        self.provide_data = [('data', (batch_size, self.default_bucket_key)), ('sequence_length', (batch_size,))]\
                            + init_states
        # specify the shape of label
        self.provide_label = [('softmax_label', (self.batch_size, self.default_bucket_key))]

    def make_data_iter_plan(self):
        "make a random data iteration plan"
        # truncate each bucket into multiple of batch-size
        bucket_n_batches = []
        self.num_batches = 0
        # for each bucket
        for i in range(len(self.data)):
            # count how many batches there are in each bucket
            bucket_n_batches.append(len(self.data[i]) / self.batch_size)
            self.num_batches += bucket_n_batches[-1]
            # throw away remaining samples that are not enough for a batch
            self.data[i] = self.data[i][:int(bucket_n_batches[i] * self.batch_size)]
        # concatenate in horizontal direction
        # bucket_plan maps each batch to its correspoding bucket index
        # suppose that batches are [1,3,2], then bucket_plan is [0,1,1,1,2,2]
        bucket_plan = np.hstack([np.zeros(n, int) + i for i, n in enumerate(bucket_n_batches)])
        # shuffle buckets, and then shuffle seqs in each bucket
        np.random.shuffle(bucket_plan)
        # shuffle all indices in each bucket
        bucket_idx_all = [np.random.permutation(len(x)) for x in self.data]

        self.bucket_plan = bucket_plan
        self.bucket_idx_all = bucket_idx_all
        # index for current batch in each bucket
        self.bucket_curr_idx = [0 for x in self.data]

        # buffers for batch data in each bucket
        self.data_buffer = []
        self.label_buffer = []
        for i_bucket in range(len(self.data)):
            if not self.model_parallel:
                # batch_size x bucket_size of i'th bucket
                data = np.zeros((self.batch_size, self.maxlen_per_bucket[i_bucket]))
                label = np.zeros((self.batch_size, self.maxlen_per_bucket[i_bucket]))
                self.data_buffer.append(data)
                self.label_buffer.append(label)
            else:
                data = np.zeros((self.maxlen_per_bucket[i_bucket], self.batch_size))
                self.data_buffer.append(data)

        if self.model_parallel:
            # Transpose data if model parallel
            for i in range(len(self.data)):
                bucket_data = self.data[i]
                self.data[i] = np.transpose(bucket_data)

    def __iter__(self):

        for i_bucket in self.bucket_plan:
            data = self.data_buffer[i_bucket]
            i_idx = self.bucket_curr_idx[i_bucket]
            # index range for shuffled data in a batch of sentences
            # note that idx is the indices within a bucket, not global indices(of sentences)
            idx = self.bucket_idx_all[i_bucket][i_idx:i_idx + self.batch_size]
            self.bucket_curr_idx[i_bucket] += self.batch_size

            # Model parallelism
            if self.model_parallel:
                if self.data[i_bucket][:, idx].shape[1] == 0:
                    print "WARNING: detected shape " + str(self.data[i_bucket][:, idx].shape)
                    continue
                data[:] = self.data[i_bucket][:, idx]
                data_batch = ModelParallelBatch(data, self.buckets[i_bucket])
                yield data_batch

            # Data parallelism
            else:
                init_state_names = [x[0] for x in self.init_states]
                data[:] = self.data[i_bucket][idx]

                label = self.label_buffer[i_bucket]
                # In language model, our goal is to predict the next word
                label[:, :-1] = data[:, 1:]
                label[:, -1] = 0
                # sequence_length for mask layer
                sequence_length = self.sequence_length[i_bucket][idx]
                data_all = [mx.nd.array(data), mx.nd.array(sequence_length)] + self.init_state_arrays
                label_all = [mx.nd.array(label)]
                data_names = ['data', 'sequence_length'] + init_state_names
                label_names = ['softmax_label']
                data_batch = SimpleBatch(data_names, data_all, label_names, label_all,
                                         self.maxlen_per_bucket[i_bucket])
                yield data_batch

    def reset(self):
        self.bucket_curr_idx = [0 for x in self.data]

if __name__ == "__main__":
    batch_size = 32
    num_buckets = 6
    num_hidden = 100
    num_lstm_layer = 1
    init_c = [('l%d_init_c' % l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    init_h = [('l%d_init_h' % l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    init_states = init_c + init_h
    vocab = default_build_vocab("./data/ptb.train.txt")
    data_train = BucketSentenceIter("./data/ptb.train.txt", vocab,
                                    num_buckets, batch_size, init_states)
    count = 1
    for batch_ in data_train:
        print "data shape:%s, label shape:%s"%(str(batch_.data[0].shape),str(batch_.label[0].shape))
        print "batch sent len:%s"%str(batch_.data[1].asnumpy())
        count += 1
    print "total %d batches"%count
    print data_train.default_bucket_key
