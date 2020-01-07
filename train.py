# -*- coding:utf-8 -*-
import os
import tensorflow as tf
from Parameters import Parameters as pm
from data_process import read_category, get_word_id, get_word2vec, process, batch_iter, seq_length, build_vocab_vector
from model import LstmCnn


def train():

    tensorboard_dir = './tensorboard/Lstm_CNN'
    save_dir = './checkpoints/Lstm_CNN'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, 'best_validation')

    tf.summary.scalar('loss', model.loss)
    tf.summary.scalar('accuracy', model.accuracy)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)
    saver = tf.train.Saver()
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)

    print("Preparing the training data....")
    x_train, y_train = process(pm.train_filename, word_ids, cat_to_id, max_length=300)
    print("Preparing the testing data....")
    x_test, y_test = process(pm.test_filename, word_ids, cat_to_id, max_length=300)
    for epoch in range(pm.num_epochs):
        print('Epoch:', epoch+1)
        num_batchs = int((len(x_train) - 1) / pm.batch_size) + 1
        batch_train = batch_iter(x_train, y_train, batch_size=pm.batch_size)
        for x_batch, y_batch in batch_train:
            real_seq_len = seq_length(x_batch)
            feed_dict = model.feed_data(x_batch, y_batch, real_seq_len, pm.keep_prob)
            _, global_step, _summary, train_loss, train_accuracy = session.run([model.optimizer, model.global_step, merged_summary,
                                                                                model.loss, model.accuracy], feed_dict=feed_dict)
            if global_step % 100 == 0:
                test_loss, test_accuracy = model.test(session, x_test, y_test)
                print('global_step:', global_step, 'train_loss:', train_loss, 'train_accuracy:', train_accuracy,
                      'test_loss:', test_loss, 'test_accuracy:', test_accuracy)

            if global_step % num_batchs == 0:
                print('Saving Model...')
                saver.save(session, save_path, global_step=global_step)


if __name__ == "__main__":
    pm = pm
    filenames = [pm.train_filename, pm.test_filename, pm.val_filename]
    categories, cat_to_id = read_category()

    if not os.path.exists(pm.vocab_name):
        build_vocab_vector(filenames, 10000)

    word_ids = get_word_id(pm.vocab_name)
    pm.vocab_size = len(word_ids)
    pm.pre_training = get_word2vec(pm.vector_word_npz)

    model = LstmCnn()
    train()
