"""
@author: hschen0712
@description: 
@time: 2016/11/22 16:22
"""
from bucket_io import ParallelCorpusIter
from trans_utils import read_tokens, build_vocab, Perplexity, DebugMetric
from model import Translate
import cPickle as pickle
import mxnet as mx

if __name__ == "__main__":
    _, word_lst = read_tokens("data/english8k.txt")
    _, word_lst = read_tokens("data/chinese8k.txt")
    src_vocab = pickle.load(open("data/chn.vcb.pkl", "r"))
    trg_vocab = pickle.load(open("data/eng.vcb.pkl", "r"))
    src_vocab_size = len(src_vocab)
    batch_size = 32
    num_gru_layer = 1
    num_hidden = 100
    num_embed = 50
    num_epoch = 1
    learning_rate = 0.01
    momentum = 0.0
    enc_init_states = [('l%d_init_h' % l, (batch_size, num_hidden)) for l in range(num_gru_layer)]
    enc_init_state_names = [x[0] for x in enc_init_states]
    data_train = ParallelCorpusIter("data/chinese8k.txt", "data/english8k.txt", src_vocab, trg_vocab,
                                    enc_init_states, batch_size)
    # define computational graph
    def sym_gen((enc_len, dec_len)):
        translate = Translate(num_gru_layer, src_vocab_size,
                 num_hidden, num_embed, use_mask=True, dropout=0.)
        data_names = ['src_data', 'src_mask', 'trg_mask'] + enc_init_state_names
        label_names = ['trg_data']
        res = translate.forward(enc_len, dec_len)
        return (res, data_names, label_names)

    contexts = [mx.context.gpu(i) for i in range(1)]

    model = mx.mod.BucketingModule(sym_gen, default_bucket_key=data_train.default_bucket_key,
                                 context=contexts)

    import logging

    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)

    model.fit(data_train,
              # eval_data=data_val,
              num_epoch=num_epoch,
              eval_metric=mx.metric.np(Perplexity),
              batch_end_callback=mx.callback.Speedometer(batch_size, 50),
              initializer=mx.init.Xavier(factor_type="in", magnitude=2.34),
              optimizer='adadelta')