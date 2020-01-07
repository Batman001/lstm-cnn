# -*- coding:utf-8 -*-
import os
import tensorflow as tf
from Parameters import Parameters as pm
from data_process import read_category, get_word_id, get_word2vec, \
    process, batch_iter, seq_length, build_vocab_vector
