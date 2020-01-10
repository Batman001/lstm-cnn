# -*- coding:utf-8- -*-
import tensorflow as tf
from tensorflow import keras
from Parameters import Parameters as pm
from data_process import batch_iter


class BiLSTM(object):
    def __init__(self):
        self.input_x = tf.placeholder(tf.int32, [None, pm.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, pm.num_classes], name='input_y')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.l2loss = tf.constant(0.0)
        self.bi_lstm()

    def bi_lstm(self):
        with tf.device("/cpu:0"), tf.name_scope('embedding'):
            # 利用预训练的词向量初始化词嵌入矩阵
            self.embedding = tf.get_variable('embedding', shape=[pm.vocab_size, pm.embedding_dim], dtype=tf.float32)
            # 利用词嵌入矩阵将输入数据中的词转化成词向量，维度[batch_size, seq_length, embedding_dim]
            embedding_input = tf.nn.embedding_lookup(self.embedding, self.input_x)

        with tf.name_scope('Bi-LSTM'):
            for idx, hidden_size in enumerate(pm.hidden_sizes):
                with tf.name_scope('Bi-LSTM' + str(idx)):
                    # 定义前向LSTM结构
                    lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(
                        num_units=hidden_size, state_is_tuple=True), output_keep_prob=self.dropout_keep_prob)
                    # 定义反向LSTM结构
                    lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(
                        num_units=hidden_size, state_is_tuple=True), output_keep_prob=self.dropout_keep_prob)
                    # 使用动态rnn 可以动态的输入序列的长度 若没有输入 则为序列的全部长度
                    # outputs是一个元祖(output_fw, output_bw)，
                    # 其中两个元素的维度都是[batch_size, max_time, hidden_size],fw和bw的hidden_size一样
                    # self.current_state 是最终的状态，二元组(state_fw, state_bw)，state_fw=[batch_size, s]，s是一个元祖(h, c)

                    outputs, self.current_state = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell,
                                                    embedding_input, scope='bi-lstm' + str(idx), dtype=tf.float32)
                    # 对outputs中的fw_cell和bw_cell的结果进行拼接 [batch_size, time_step, hidden_size*2]
                    self.outputs_ = tf.concat(outputs, 2)

            # 去除最后时间步的输出作为全连接的输入
            final_output = self.outputs_[:, -1, :]
            # 因为是双向LSTM，最终的输出值是fw和bw的拼接，因此要乘以2
            output_size = pm.hidden_sizes[-1] * 2
            # reshape 后的全连接层的输入维度
            output = tf.reshape(final_output, [-1, output_size])

        with tf.name_scope('output'):
            output_W = tf.Variable(tf.truncated_normal([output_size, pm.num_classes], stddev=0.1),name='output_W')
            output_b = tf.Variable(tf.constant(0.1, shape=[pm.num_classes]), name='output_b')
            self.l2loss += tf.nn.l2_loss(output_W)
            self.l2loss += tf.nn.l2_loss(output_b)

            self.logits = tf.nn.xw_plus_b(output, output_W, output_b, name='logits')
            self.predictions = tf.argmax(tf.nn.softmax(self.logits, 1), name='predictions')

        with tf.name_scope('loss'):
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(self.input_x, 1),
                                                                           logits=self.logits)
            self.loss = tf.reduce_mean(cross_entropy)

        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate=pm.learning_rate)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            gradients, _ = tf.clip_by_global_norm(gradients, pm.clip)
            self.optimizer = optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)

        with tf.name_scope('accuracy'):
            self.correct_pred = tf.equal(tf.argmax(self.predictions, 1), tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32), name='accuracy')

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
