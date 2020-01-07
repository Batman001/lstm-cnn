# -*- coding:utf-8 -*-
import numpy as np
from lstmcnn.lstm_cnn import LstmCnn
import tensorflow as tf
from data_process import read_category,get_word_id, get_word2vec,process, batch_iter, seq_length
from Parameters import Parameters as pm


def predict():

    pre_labels = []
    labels = []
    session = tf.Session()
    save_path = tf.train.latest_checkpoint('./checkpoints/Lstm_CNN')
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)

    val_x, val_y = process(pm.val_filename, word_to_ids, cat_to_id, max_length=pm.seq_length)
    batch_val = batch_iter(val_x, val_y, batch_size=64)

    for x_batch, y_batch in batch_val:
        real_seq_len = seq_length(x_batch)
        feed_dict = model.feed_data(x_batch, y_batch, real_seq_len, 1.0)
        predict_label = session.run(model.predict, feed_dict=feed_dict)
        pre_labels.extend(predict_label)
        labels.extend(y_batch)
    return pre_labels, labels


if __name__ == "__main__":
    pm = pm
    sentences = []
    label2 = []
    categories, cat_to_id = read_category()
    word_to_ids = get_word_id(pm.vocab_name)
    pm.vocab_size = len(word_to_ids)
    pm.pre_training = get_word2vec(pm.vector_word_npz)


    model = LstmCnn()
    pre_labels, labels = predict()
    correct = np.equal(pre_labels, np.argmax(labels, 1))
    accuracy = np.mean(np.cast['float32'](correct))

    print('accuracy', accuracy)
    print('预测前十项为:', ' '.join(str(pre_labels[:10])))
    print('实际的label前十项为：', ' '.join(str(np.argmax(labels[:10], 1))))


