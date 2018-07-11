# coding:utf-8
# 基于朴素贝叶斯分类器从个人广告中获取区域倾向
# 但是由于rss源似乎有问题，一直得不到数据

import feedparser
import random
import numpy as np

from bayes import create_vocab_list, bag_of_word2vec, trainNB0, classifyNB

def text_parse(big_string):
    """
    分割文档去掉标点并返回单词长度大于2的单词列表
    """
    import re
    list_of_tokens = re.split(r'\W*', big_string)
    return [tok.lower() for tok in list_of_tokens if len(tok) > 2]

def calc_most_freq(vocab_list, full_text):
    """
    vocab_list: list, 词典，里面的单词唯一
    full_text: list, 文档中的词组成的列表
    这个函数统计文档中的词出现的次数，返回前30出现频率最高的词及次数
    然后从单词表中去掉词频前30的单词，这样可以提高分类性能
    因为语言中大部分都是冗余和结构辅助词，这个也称为停用词
    除了这样的方法也可以直接从网上下载停用词表然后去除
    """
    import operator
    freq_dict = {}
    for token in vocab_list:
        freq_dict[token] = full_text.count(token)
    sorted_freq = sorted(freq_dict.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_freq[:30]

def local_words(feed1, feed0):
    """
    feed1: rss源1
    feed0: rss源0
    """
    import feedparser
    # 文档列表
    doc_list = []
    # 文档类别列表
    class_list = []
    full_text = []
    # 计算两个源中用于训练的数据的数量
    min_len = min(len(feed1['entries']), len(feed0['entries']))
    # 遍历每个训练数据（文档）
    for i in range(min_len):
        # 分词，去掉标点符号
        word_list = text_parse(feed1['entries'][i]['summary'])
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(1)
        # 分词，去掉标点符号
        word_list = text_parse(feed0['entries'][i]['summary'])
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(0)
    # 构造单词列表
    vocab_list = create_vocab_list(doc_list)
    # 统计出出现频率前30的单词
    top30_words = calc_most_freq(vocab_list, full_text)
    # 从单词中去掉高频词汇
    for pair_w in top30_words:
        if pair_w[0] in vocab_list:
            vocab_list.remove(pair_w[0])
    # 因为是两个源的数据所以*2
    training_set = range(2*min_len)
    #print(training_set)
    # 接下来构造测试数据集
    test_set = []
    for i in range(20):
        # 随机从训练数据集中获得20个数据，同时在训练数据集中将其删除
        rand_index = int(random.uniform(0, len(training_set)))
        test_set.append(training_set[rand_index])
        del(training_set[rand_index])
    train_mat = []
    train_classes = []
    # 下面遍历training_set构造出最终用于训练的文档向量和标签
    for doc_index in training_set:
        # 用词袋模型构造每个文档的向量
        train_mat.append(bag_of_word2vec(vocab_list, doc_list[doc_index]))
        train_classes.append(class_list[doc_index])
    # 训练贝叶斯分类器，这里由于使用两源同样数量的数据，所以p_spam为0.5
    p0_v, p1_v, p_spam = trainNB0(np.array(train_mat), np.array(train_classes))
    error_count = 0
    # 测试数据
    for doc_index in test_set:
        # 构造测试文档的向量
        word_vec = bag_of_word2vec(vocab_list, doc_list[doc_index])
        if classifyNB(np.array(word_vec), p0_v, p1_v, p_spam) != class_list[doc_index]:
            error_count += 1
        print('the error rate is: ', float(error_count)/len(test_set))
        return vocab_list, p0_v, p1_v

def get_top_words(ny, sf):
    """
    ny: rss源
    sf: rss源
    统计两个源中出现次数最多的单词，就是这两个地区的人使用最多的词汇
    """
    import operator
    vocab_list, p0_v, p1_v = local_words(ny, sf)
    top_ny = []
    top_sf = []
    for i in range(len(p0_v)):
        if p0_v[i] > -6.0:
            top_sf.append((vocab_list[i], p0_v[i]))
        if p1_v[i] > -6.0:
            top_ny.append((vocab_list[i], p1_v[i]))
    sorted_sf = sorted(top_sf, key=lambda pair: pair[i], reverse=True)
    print('sf:')
    for item in sorted_sf:
        print(item[0])
    sorted_ny = sorted(top_ny, key=lambda pair:pair[0], reverse=True)
    print('ny:')
    for item in sorted_ny:
        print(item[0])

def main():
    # 获取rss源
    ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
    sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
    vocab_list, p_sf, p_ny = local_words(ny, sf)
    get_top_words(ny, sf)

if __name__ == '__main__':
    main()