# -*- coding:utf-8 -*-
import tensorflow as tf
from data_process import batch_iter, seq_length
from Parameters import Parameters as pm


class LstmCnn(object):

    def __init__(self):
        self.input_x = tf.placeholder(tf.int32, shape=[None, pm.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, shape=[None, pm.num_classes], name='input_y')
        self.length = tf.placeholder(tf.int32, shape=[None], name='rnn_length')
        self.keep_pro = tf.placeholder(tf.float32, name='dropout')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.lstm_cnn()

    def lstm_cnn(self):

        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            self.embedding = tf.get_variable('embedding', shape=[pm.vocab_size, pm.embedding_dim],
                                             initializer=tf.constant_initializer(pm.pre_training))
            embedding_input = tf.nn.embedding_lookup(self.embedding, self.input_x)

        with tf.name_scope('LSTM'):
            cell = tf.nn.rnn_cell.LSTMCell(pm.hidden_dim, state_is_tuple=True)
            Cell = tf.contrib.rnn.DropoutWrapper(cell, pm.keep_prob)
            output, _ = tf.nn.dynamic_rnn(cell=Cell, inputs=embedding_input,
                                          sequence_length=self.length, dtype=tf.float32)
            print("LSTM的输出层：", output)

        with tf.name_scope('CNN'):
            outputs = tf.expand_dims(output, -1)  # [batch_size,seq_length, hidden_dim, 1]
            print("测试经过expand的dim为：", outputs)
            pooled_outputs = []
            for i, filter_size in enumerate(pm.filter_sizes):
                filter_shape = [filter_size, pm.hidden_dim, 1, pm.num_filters]
                w = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='w')
                # 每一个filters进行bias的添加 卷积操作
                b = tf.Variable(tf.constant(0.1, shape=[pm.num_filters]), name='b')
                conv = tf.nn.conv2d(outputs, w, strides=[1, 1, 1, 1], padding='VALID', name='conv')
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')

                # 池化操作
                pooled = tf.nn.max_pool(h, ksize=[1, pm.seq_length-filter_size+1, 1, 1],
                                        strides=[1,1,1,1], padding='VALID', name='pool')

                pooled_outputs.append(pooled)
            output_ = tf.concat(pooled_outputs, 3)
            print("CNN层的输出1为:", output_)
            # 拉平池化之后的矩阵 用于连接全连接层
            self.output = tf.reshape(output_, shape=[-1, len(pm.filter_sizes)*pm.num_filters])
            print("CNN层的输出2为:", self.output)

        with tf.name_scope('output'):
            out_final = tf.nn.dropout(self.output, keep_prob=self.keep_pro)
            # 全连接层
            o_w = tf.Variable(tf.truncated_normal([len(pm.filter_sizes)*pm.num_filters, pm.num_classes], stddev=0.1), name='o_w')
            o_b = tf.Variable(tf.constant(0.1, shape=[pm.num_classes]), name='o_b')
            self.logits = tf.matmul(out_final, o_w) + o_b
            self.predict = tf.argmax(tf.nn.softmax(self.logits), 1, name='score')

        with tf.name_scope('loss'):
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(self.input_y,1),
                                                                           logits=self.logits)
            self.loss = tf.reduce_mean(cross_entropy)

        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(pm.learning_rate)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            # 让权重的更新限制在一个合适的范围
            gradients, _ = tf.clip_by_global_norm(gradients, pm.clip)
            self.optimizer = optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)

        with tf.name_scope('accuracy'):
            correct = tf.equal(self.predict, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')

    def feed_data(self, x_batch, y_batch, real_seq_len, keep_pro):
        feed_dict = {self.input_x: x_batch,
                     self.input_y: y_batch,
                     self.length: real_seq_len,
                     self.keep_pro: keep_pro}
        return feed_dict

    def test(self, sess, x, y):
        global test_loss, test_accuracy
        batch_test = batch_iter(x, y, batch_size=pm.batch_size)
        for x_batch, y_batch in batch_test:
            real_seq_len = seq_length(x_batch)
            feed_dict = self.feed_data(x_batch, y_batch, real_seq_len, 1.0)
            test_loss, test_accuracy = sess.run([self.loss, self.accuracy], feed_dict=feed_dict)

        return test_loss, test_accuracy
