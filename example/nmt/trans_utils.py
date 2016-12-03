"""
@author: hschen0712
@description: utils for translate
@time: 2016/11/22 16:46
"""
from collections import OrderedDict
import nltk
import cPickle

tokenize = lambda x: x.strip().split(" ")


def read_tokens(path):
    sents = []
    word_lst = []
    with open(path, "r") as ins:
        for line in ins:
            line = line.replace("\n", "").replace("\r", "").replace("\r\n", "")
            sents.append(line)
            word_lst += tokenize(line)
    return sents, word_lst


def build_vocab(word_lst, vocab_size, vocab_path):
    word_freq = nltk.FreqDist(word_lst)
    vocab = OrderedDict(word_freq.most_common())
    # don't worry if len(vocab)<vocab_size
    words = vocab.keys()[:vocab_size-3]
    word2idx = OrderedDict([(w, i+2) for i, w in enumerate(words)])
    word2idx["<s>"] = 0
    word2idx["<unk>"] = 1
    word2idx["</s>"] = len(word2idx)
    cPickle.dump(word2idx, open(vocab_path, "w"))
    # return actual vocab size


def text2id(sentence, the_vocab):
    words = sentence.split(' ')
    words = [the_vocab[w] for w in words if len(w) > 0]
    return words
