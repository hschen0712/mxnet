"""
@author: hschen0712
@description: 
@time: 2016/12/03 21:43
"""
import mxnet as mx
from gru import gru, GRUParam, GRUState

class Encoder(object):
    def __init__(self, num_gru_layer, src_vocab_size,
                 num_hidden, num_embed, use_mask=True, dropout=0.):
        self.num_gru_layer = num_gru_layer
        self.vocab_size = src_vocab_size
        self.num_hidden = num_hidden
        self.num_embed = num_embed
        self.use_mask = use_mask
        self.dropout = dropout
        self.embed_weight = mx.sym.Variable("embed_weight")
        param_cells = []
        last_states = []
        for i in range(num_gru_layer):
            param_cells.append(GRUParam(gates_i2h_weight=mx.sym.Variable("l%d_i2h_gates_weight" % i),
                                        gates_i2h_bias=mx.sym.Variable("l%d_i2h_gates_bias" % i),
                                        gates_h2h_weight=mx.sym.Variable("l%d_h2h_gates_weight" % i),
                                        gates_h2h_bias=mx.sym.Variable("l%d_h2h_gates_bias" % i),
                                        trans_i2h_weight=mx.sym.Variable("l%d_i2h_trans_weight" % i),
                                        trans_i2h_bias=mx.sym.Variable("l%d_i2h_trans_bias" % i),
                                        trans_h2h_weight=mx.sym.Variable("l%d_h2h_trans_weight" % i),
                                        trans_h2h_bias=mx.sym.Variable("l%d_h2h_trans_bias" % i)))
            state = GRUState(h=mx.sym.Variable("l%d_init_h" % i))
            last_states.append(state)
        self.param_cells = param_cells
        self.last_states = last_states
        assert (len(last_states) == num_gru_layer)

    def forward(self, seq_len):
        # batch_size x seq_len
        data = mx.sym.Variable('src_data')
        # data: batch_size x seq_len
        # input_size stands for vocab_size
        # ouput shape: batch_size x seq_len x vocab_size
        embed = mx.sym.Embedding(data=data, input_dim=self.vocab_size,
                                 weight=self.embed_weight, output_dim=self.num_embed, name='src_embed')
        # wordvec: can be think of as a list of length seq_len
        # that contains tensors of shape batch_size x vocab_size
        wordvec = mx.sym.SliceChannel(data=embed, num_outputs=seq_len, squeeze_axis=1)
        if self.use_mask:
            mask = mx.sym.Variable('src_mask')
            maskvec = mx.sym.SliceChannel(data=mask, num_outputs=seq_len)

        hidden_all = []
        for seqidx in range(seq_len):
            hidden = wordvec[seqidx]
            # stack GRU
            for i in range(self.num_gru_layer):
                if i == 0:
                    dp_ratio = 0.
                else:
                    dp_ratio = self.dropout
                next_state = gru(self.num_hidden, indata=hidden,
                                 prev_state=self.last_states[i],
                                 param=self.param_cells[i],
                                 seqidx=seqidx, layeridx=i, dropout=dp_ratio)
                if self.use_mask:
                    _mask = maskvec[seqidx]
                    # apply mask
                    next_h = mx.sym.broadcast_mul(1. - _mask, self.last_states[i].h) \
                             + mx.sym.broadcast_mul(_mask, next_state.h)
                    next_state = GRUState(h=next_h)
                hidden = next_state.h
                self.last_states[i] = next_state
            # decoder
            if self.dropout > 0.:
                hidden = mx.sym.Dropout(data=hidden, p=self.dropout)
            hidden_all.append(hidden)

        # return last state
        return self.last_states

class Decoder(object):
    def __init__(self, num_gru_layer, trg_vocab_size,
                 num_hidden, num_embed, use_mask=True, dropout=0.):
        self.num_gru_layer = num_gru_layer
        self.num_class = trg_vocab_size
        self.num_hidden = num_hidden
        self.num_embed = num_embed
        self.use_mask = use_mask
        self.dropout = dropout
        self.embed_weight = mx.sym.Variable("trg_embed_weight")
        self.cls_weight = mx.sym.Variable("cls_weight")
        self.cls_bias = mx.sym.Variable("cls_bias")
        param_cells = []
        for i in range(num_gru_layer):
            param_cells.append(GRUParam(gates_i2h_weight=mx.sym.Variable("dec_l%d_i2h_gates_weight" % i),
                                        gates_i2h_bias=mx.sym.Variable("dec_l%d_i2h_gates_bias" % i),
                                        gates_h2h_weight=mx.sym.Variable("dec_l%d_h2h_gates_weight" % i),
                                        gates_h2h_bias=mx.sym.Variable("dec_l%d_h2h_gates_bias" % i),
                                        trans_i2h_weight=mx.sym.Variable("dec_l%d_i2h_trans_weight" % i),
                                        trans_i2h_bias=mx.sym.Variable("dec_l%d_i2h_trans_bias" % i),
                                        trans_h2h_weight=mx.sym.Variable("dec_l%d_h2h_trans_weight" % i),
                                        trans_h2h_bias=mx.sym.Variable("dec_l%d_h2h_trans_bias" % i)))
        self.param_cells = param_cells

    def forward(self, dec_len, enc_states):
        # batch_size x seq_len
        data = mx.sym.Variable('trg_data')

        embed = mx.sym.Embedding(data=data, input_dim=self.num_class,
                                 weight=self.embed_weight, output_dim=self.num_embed, name='trg_embed')
        # wordvec: can be think of as a list of length seq_len
        # that contains tensors of shape batch_size x num_embed
        wordvec = mx.sym.SliceChannel(data=embed, num_outputs=dec_len, squeeze_axis=1)
        if self.use_mask:
            mask = mx.sym.Variable('trg_mask')
            maskvec = mx.sym.SliceChannel(data=mask, num_outputs=dec_len)

        hidden_all = []
        last_states = []
        for i in range(self.num_gru_layer):
            last_states.append(enc_states[i])
        self.last_states = last_states
        assert (len(last_states) == self.num_gru_layer)

        for seqidx in range(dec_len):
            # batch_size x num_embed
            hidden = wordvec[seqidx]
            # stack GRU
            for i in range(self.num_gru_layer):
                if i == 0:
                    dp_ratio = 0.
                else:
                    dp_ratio = self.dropout
                next_state = gru(self.num_hidden, indata=hidden,
                                 prev_state=self.last_states[i],
                                 param=self.param_cells[i],
                                 seqidx=seqidx, layeridx=i, dropout=dp_ratio)
                if self.use_mask:
                    _mask = maskvec[seqidx]
                    # apply mask
                    next_h = mx.sym.broadcast_mul(1. - _mask, self.last_states[i].h) \
                             + mx.sym.broadcast_mul(_mask, next_state.h)
                    next_state = GRUState(h=next_h)
                hidden = next_state.h
                self.last_states[i] = next_state
            # decoder
            if self.dropout > 0.:
                hidden = mx.sym.Dropout(data=hidden, p=self.dropout)
            hidden_all.append(hidden)
        # (dec_len * batch_size) x num_embed
        hidden_concat = mx.sym.Concat(*hidden_all, dim=0)
        # pred: (dec_len * batch_size) x num_class
        pred = mx.sym.FullyConnected(data=hidden_concat, num_hidden=self.num_class,
                                     weight=self.cls_weight, bias=self.cls_bias, name='pred')

        label = mx.sym.transpose(data=data)
        label = mx.sym.Reshape(data=label, target_shape=(0,))

        # return last state
        return mx.sym.SoftmaxOutput(data=pred, label=label, name='softmax')

class Translate(object):
    def __init__(self, num_gru_layer, src_vocab_size,
                 num_hidden, num_embed, use_mask=True, dropout=0.):
        self.encoder = Encoder(num_gru_layer, src_vocab_size,
                 num_hidden, num_embed, use_mask, dropout)
        self.decoder = Decoder(num_gru_layer, src_vocab_size,
                 num_hidden, num_embed, use_mask, dropout)

    def forward(self, enc_len, dec_len):
        enc_states = self.encoder.forward(enc_len)

        return self.decoder.forward(dec_len, enc_states)


