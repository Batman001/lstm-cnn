# -*- coding:utf-8 -*-
from collections import Counter
import tensorflow as tf
from tensorflow import keras
import numpy as np
import codecs
import jieba
import re


def read_file(filename):
    re_han = re.compile(u"([\u4E00-\u9FD5a-zA-Z0-9+#&\._%]+)")
    contents, labels = [], []
    with codecs.open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            row = 1
            try:
                line = line.rstrip()
                assert len(line.split("\t") == 2)
                label, content =line.split('\t')
                labels.append(label)
                blocks = re_han.split(content)
                word = []

                for block in blocks:
                    if re_han.match(block):
                        cut_res = jieba.lcut(block)
                        word.extend(cut_res)
                print("切割第" + str(row) + "行的结果为:", word)
                contents.append(word)
                row += 1
            except:
                pass
    return labels, contents


def build_vocab_vector(filenames, voc_size=10000):
    """
    去停用词 得到前9999个词 获取对应词的以及其词向量
    并写入本地磁盘
    vocab_word.txt 全部9999个词
    vector_word.npz 全部9999个词的100为词向量
    :param filenames:
    :param voc_size:
    :return:
    """
    stop_words = codecs.open('./data/stopwords.txt', 'r', encoding='utf-8')
    stop =[key.strip(' \n') for key in stop_words]

    all_data = []
    j = 1
    # 每一个词的维度为100维
    embeddings = np.zeros([10000, 100])

    for filename in filenames:
        labels, contents = read_file(filename)
        for each_line in contents:
            line = []
            for w_index in range(len(each_line)):
                # 去停用词
                if(str(each_line[w_index]) not in stop):
                    line.append(each_line[w_index])

            all_data.extend(line)

    counter = Counter(all_data)
    counter_pairs = counter.most_common(voc_size - 1)
    word, _ = list(zip(*counter_pairs))

    f = codecs.open('./data/vector_word.txt', 'r', encoding='utf-8')
    vocab_word = codecs.open('./data/vocab_word.txt', 'w', encoding='utf-8')

    for each_line in f:
        item = each_line.split(' ')
        key = item[0]
        vec = np.array(item[1:], dtype='float32')
        if key in word:
            embeddings[j] = np.array(vec)
            vocab_word.write(key.strip('\r') + '\n')
            j += 1
    np.savez_compressed('/data/vector_word.npz', embeddings=embeddings)
    f.close()
    vocab_word.close()


def get_word_id(filename):
    key = codecs.open(filename, 'r', encoding='utf-8')
    word_id = {'<PAD>': 0}
    w_index = 1
    for w in key:
        w = w.strip('\r').strip('\n')
        word_id[w] = w_index
        w_index += 1
    return word_id


def read_category():
    categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
    cat_to_id = dict(zip(categories, range(len(categories))))
    return categories, cat_to_id


def process(filename, word_to_id, cat_to_id, max_length=300):
    """
    对filename的文本文件进行预处理的流程程序入口
    :param filename:
    :param word_to_id:
    :param cat_to_id:
    :param max_length:
    :return:
    """
    labels, contents = read_file(filename)

    data_id, label_id = [], []

    for i in range(len(contents)):
        data_id.append(word_to_id[x] for x in contents[i] if x in word_to_id)
        label_id.append(cat_to_id[labels[i]])

    x_pad = keras.preprocessing.sequence.pad_sequences(data_id, max_length,
                                                       padding='post', truncating='post')
    y_pad = keras.utils.to_categorical(label_id)

    return x_pad, y_pad


def get_word_to_vec(filename):
    with np.load(filename) as data:
        return data['embeddings']


def batch_iter(x, y, batch_size=64):
    data_len = len(x)
    num_batch = int(data_len-1) // batch_size + 1
    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i+1)*batch_size, data_len)
        yield x_shuffle[start_id: end_id], y_shuffle[start_id, end_id]


def seq_length(x_batch):
    real_seq_len = []
    for line in x_batch:
        real_seq_len.append(np.sign(line))

    return real_seq_len
