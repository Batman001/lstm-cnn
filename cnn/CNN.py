# -*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow import keras
from Parameters import Parameters as pm
from data_process import batch_iter


class TextCNN(object):
    def __init__(self):
        self.input_x = tf.placeholder(tf.int32, [None, pm.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, pm.num_classes], name='input_y')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        # 定义L2损失
        self.l2loss = tf.constant(0.0)
        self.cnn()

    def cnn(self):
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            # 利用预训练的词向量初始化词嵌入矩阵
            self.embedding = tf.get_variable('embedding', shape=[pm.vocab_size, pm.embedding_dim], dtype=tf.float32)
            embedding_input = tf.nn.embedding_lookup(self.embedding, self.input_x)
            embedding_expand_input = tf.expand_dims(embedding_input, -1)

        with tf.name_scope('conv'):
            pooled_outputs = []
            for i, filter_size in enumerate(pm.filter_sizes):
                with tf.name_scope('conv-maxpool-%s' %filter_size):
                    # 卷积层 卷积核尺寸为 filter_size * embedding_dim 卷积核个数为 pm.num_filters
                    filter_shape = [filter_size, pm.embedding_dim, 1, pm.num_filters]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
                    b = tf.Variable(tf.constant(0.1, shape=[pm.num_filters]), name='b')
                    conv = tf.nn.conv2d(embedding_expand_input, W, strides=[1,1,1,1], padding='VALID', name='conv')
                    # relu函数的非线性映射
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')

                    # 池化层  最大池化 池化是对卷积后序列取最大值
                    # k_size shape: [batch, height, width, channels]
                    pooled = tf.nn.max_pool(h, ksize=[1, pm.seq_length-filter_size + 1, 1, 1],
                                            strides=[1, 1, 1, 1], padding='VALID', name='max_pool')
                    # 将三种size的filter输出一起添加到列表中
                    pooled_outputs.append(pooled)

            # 得到CNN网络的输出长度
            num_filters_total = pm.num_filters * len(pm.filter_sizes)
            # 池化后维度不变 按照最后维度channel 来 concat
            self.h_pool = tf.concat(pooled_outputs, 3)

            # 摊平成二维的数据 输入到全连接层
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        with tf.name_scope('dropout'):
            self.h_dropout = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # 全连接层的输出
        with tf.name_scope('fc_output'):
            output_W = tf.Variable(tf.truncated_normal([len(pm.filter_sizes) * pm.num_filters, pm.num_classes], stddev=0.1),
                              name='output_W')
            output_b = tf.Variable(tf.constant(0.1, shape=[pm.num_classes]), name='output_b')
            self.l2loss += tf.nn.l2_loss(output_W)
            self.l2loss += tf.nn.l2_loss(output_b)

            self.logits = tf.nn.xw_plus_b(self.h_dropout, output_W, output_b, name='logits')
            self.predictions = tf.argmax(tf.nn.softmax(self.logits), 1, name='predictions')
            print("当前预测结果为： ", self.predictions)

        with tf.name_scope('loss'):
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(self.input_y,1),
                                                                           logits=self.logits)
            self.loss = tf.reduce_mean(cross_entropy) + pm.l2RegLambda * self.l2loss

        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(pm.learning_rate)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            # 让权重的更新限制在一个合适的范围
            gradients, _ = tf.clip_by_global_norm(gradients, pm.clip)
            self.optimizer = optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)

        with tf.name_scope('accuracy'):
            correct = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')

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














