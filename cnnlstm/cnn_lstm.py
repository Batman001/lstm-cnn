# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf
from data_process import batch_iter
from Parameters import Parameters as pm


class CnnLstm(object):
    def __init__(self):
        self.input_x = tf.placeholder(tf.int32, [None, pm.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, pm.num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.l2loss=tf.constant(0.0)
        self.cnn_lstm()

    def cnn_lstm(self):
        # embedding layer
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            self.embedding = tf.get_variable('embedding', shape=[pm.vocab_size, pm.embedding_dim],
                                             initializer=tf.constant_initializer(pm.pre_training))
            embedding_input = tf.nn.embedding_lookup(self.embedding, self.input_x)
            embedding_input_expand = tf.expand_dims(embedding_input, -1)
            print("embedding层输入的shape为：", embedding_input_expand)

        # convolution layer + max pooling layer(per filter)
        with tf.name_scope('CNN'):
            pooled_outputs = []
            for i, filter_size in enumerate(pm.filter_sizes):
                filter_shape = [filter_size, pm.embedding_dim, 1, pm.num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
                b = tf.Variable(tf.constant(0.1, shape=[pm.num_filters]), name='b')
                conv = tf.nn.conv2d(embedding_input_expand, W, strides=[1,1,1,1], padding='VALID', name='conv')

                # relu 非线性激活函数
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')

                # 池化操作 max pooling
                pooled = tf.nn.max_pool(h, ksize=[1,pm.seq_length-filter_size+1,1,1],
                                        strides=[1,1,1,1], padding='VALID', name='max_pool')
                pooled_outputs.append(pooled)

            # combining pooled features
            num_filter_total = pm.num_filters * len(pm.filter_sizes)
            self.h_pool = tf.concat(pooled_outputs, 3)
            print("CNN的输出层为：", self.h_pool)
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filter_total])
            print("CNN的输出层拉平为：", self.h_pool_flat)

        # dropout layer
        with tf.name_scope('dropout'):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # lstm layer
        with tf.name_scope('LSTM'):
            cell = tf.nn.rnn_cell.LSTMCell(pm.hidden_dim, state_is_tuple=True)
            self.h_drop_exp = tf.expand_dims(self.h_drop, -1)
            val, state = tf.nn.dynamic_rnn(cell=cell, inputs=self.h_drop_exp, dtype=tf.float32)

            val2 = tf.transpose(val, [1, 0, 2])
            last = tf.gather(val2, int(val2.get_shape()[0])-1)

            out_weight = tf.Variable(tf.random_normal([pm.hidden_dim, pm.num_classes]))
            out_bias = tf.Variable(tf.random_normal([pm.num_classes]))

        with tf.name_scope('output'):
            self.logits = tf.nn.xw_plus_b(last, out_weight, out_bias, name='scores')
            self.predictions = tf.nn.softmax(self.logits, name='predictions')

        with tf.name_scope('loss'):
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(self.input_y, 1),
                                                                           logits=self.logits)
            self.loss = tf.reduce_mean(cross_entropy, name='loss')

        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdadeltaOptimizer(pm.learning_rate)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            # 让权重的更新限制在一个合适的范围
            gradients, _ = tf.clip_by_global_norm(gradients, pm.clip)
            self.optimizer = optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)

        with tf.name_scope('accuracy'):
            self.correct_pred = tf.equal(tf.argmax(self.predictions, 1), tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, 'float'), name='accuracy')

    def feed_data(self, x_batch, y_batch, keep_pro):
        feed_dict = {self.input_x: x_batch,
                     self.input_y: y_batch,
                     self.dropout_keep_prob: keep_pro}
        return feed_dict


    def test(self, sess, x, y):
        global test_loss, test_accuracy
        batch_test = batch_iter(x, y, batch_size=pm.batch_size)
        for x_batch, y_batch in batch_test:
            feed_dict = self.feed_data(x_batch, y_batch, 1.0)
            test_loss, test_accuracy = sess.run([self.loss, self.accuracy], feed_dict=feed_dict)

        return test_loss, test_accuracy
















