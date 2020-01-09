# -*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow import keras
from Parameters import Parameters as pm
from data_process import batch_iter


class TextCnn(object):
    def __init__(self):
        self.input_x = tf.placeholder(tf.int32, [None, pm.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, pm.num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.cnn()

    def cnn(self):
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            self.embedding = tf.get_variable('embedding', shape=[pm.vocab_size, pm.embedding_dim])
            embedding_input = tf.nn.embedding_lookup(self.embedding, self.input_x)

        with tf.name_scope('CNN'):
            # CNN layer
            conv = tf.layers.conv1d(embedding_input, pm.num_filters, kernel_size=5, name='conv')
            # global max pooling layer
            gmp = tf.reduce_mean(conv, reduction_indices=[1], name='gmp')

        with tf.name_scope('score'):
            fc = tf.layers.dense(gmp, pm.hidden_dim, name='fc1')
            fc = tf.nn.dropout(fc, self.dropout_keep_prob)
            fc = tf.nn.relu(fc)
            # 分类器
            self.logits = tf.layers.dense(fc, pm.num_classes, name='fc2')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)

        with tf.name_scope('optimizer'):
            # 损失函数 交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            # 优化器
            self.optimizer = tf.train.AdamOptimizer(learning_rate=pm.learning_rate).minimize(self.loss)

        with tf.name_scope('accuracy'):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

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
