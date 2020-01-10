# -*- coding: utf-8 -*-
class Parameters(object):

    embedding_dim = 100
    vocab_size = 10000
    pre_training = None

    seq_length = 300
    num_classes = 10
    hidden_dim = 128
    filter_sizes = [2, 3, 4]
    num_filters = 128

    keep_prob = 0.5
    learning_rate = 1e-3
    # learning rate decay
    lr_decay = 0.9
    # gradient clipping threshold
    clip = 9.0

    num_epochs = 3
    batch_size = 64

    l2RegLambda = 0.02

    # LSTM hyper parameters
    hidden_sizes = [256, 256]  # 单层LSTM结构的神经元个数

    # train data
    train_filename = './data/cnews.train.txt'
    # test data
    test_filename = './data/cnews.test.txt'
    # val data
    val_filename = './data/cnews.val.txt'
    # vocabulary
    vocab_name = './data/vocab_word.txt'
    # vector_word trained by word2vec
    vector_word_filename = './data/vector_word.txt'
    # save vector_word to numpy file
    vector_word_npz = './data/vector_word.npz'
