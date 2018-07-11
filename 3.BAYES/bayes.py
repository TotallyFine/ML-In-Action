# coding:utf-8
# 朴素贝叶斯方法进行分类，不会直接给出类别只会给出概率
# 在数据较少的情况下仍然有效，可以处理多类别问题
# 对输入数据的准备方式比较敏感

# 有大量特征时，绘制特征作用不大，此时使用直方图效果更好

# 如果每个特征需要N个样本，那么10个特征就需要N**10个样本
# 对于包含1000个特征的词汇表将需要N**1000个样本
# 所需要的样本数会随着特征数目增大而迅速增长
# 但是如果特征之间相互独立，那么样本数就可以减少到1000*N
# 当然实际中并不是这个样子的，而朴素贝叶斯就直接当成了独立
# 这也是朴素的原因，朴素贝叶斯的另一个假设是每个特征同等重要

# 这份代码实现的功能是根据一个文档中含有的单词来判断这个文档是否是侮辱性言论
# ci是判断出的类别，w1-n是文档中含有这个单词的概率
# p(ci | w1, w2,...wn) = p(w1,w2,...wn|ci) * p(ci) / p(w1,w2,...wn)
# p(ci) 直接统计文档得出
# 因为朴素贝叶斯直接假设每个特征是相互独立的所以得到下式
# p(w1, w2, ..., wn | ci) = p(w1|ci) * p(w2 | ci) *...*p(wn | ci)
# 
# 统计类别为ci的文档中w1-wn出现的次数即可得到p(wi, ci)
# 从而得到p(w1, w2,...,wn | ci) = p(w1|ci) * p(w2 | ci) *...*p(wn | ci) = p(w1,ci)/p(ci) * p(w2|ci)/p(ci) * ...
#
# 具体的操作如下：
# wi ci的取值都是1或0，那么对单词i来说p(wi=0, ci=0) + p(wi=1,ci=0) + p(wi=0,ci=1) + p(wi=1,ci=1)=1
# 那么p(wi=0 | ci=0) = p(wi=0, ci=0)/p(ci=0), 那么p(wi=1|ci=0) = 1 - p(wi=0|ci=0)
# 遍历文档的时候设置计数器，记录ci=0以及ci=1的个数
# 并统计当ci=0时wi=1，以及ci=1时wi=1的个数，最后(ci=0时wi=1次数)/(ci=0的次数)=p(wi=1|ci=0)

import numpy as np

def load_dataset():
    posting_list = [['my', 'dog', 'has', 'flea',
                     'problems', 'help', 'please'],
                    ['maybe', 'not', 'take', 'him',
                     'to', 'dog', 'park', 'stupid'],
                    ['my', 'dalmation', 'is', 'so', 'cute',
                     'I', 'love', 'him'],
                    ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                    ['mr', 'licks', 'ate', 'my', 'steak', 'how',
                     'to', 'stop', 'him'],
                    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_vec = [0, 1, 0, 1, 0, 1] # 1代表这个文档中含有侮辱性文字 0代表正常言论
    return posting_list, class_vec

def create_vocab_list(dataset):
    """
    dataset: list, 数据集 二维，dataset[i]是第i条数据，dataset[i][0]第i个数据的第0个特征
    这个函数将单词变成唯一的集合,产生单词表，用于构造文档的向量
    由于中间使用集合进行运算，所以同一个输入，每次的输出的顺序可能不同
    """
    vocab_set = set([])
    for document in dataset:
        # 将两个字典合并，逻辑或
        vocab_set = vocab_set | set(document)
    return list(vocab_set)

def set_of_words2vec(vocab_list, input_set):
    """
    vocab_list: list, 所有文档中的单词构成的列表，没有重复
    input_set: set, 输入的单词集合,也即一个文档中的单词惟一化之后的集合
    这个函数用于判断输入的单词是否已经在构造好向量的列表中
    并据此产生这个文档的向量，若文档中有单词表中的单词则将这个文档的向量
    代表这个单词的位置置为1，否则置为0
    change a set of words to vector，a set of words means document
    这个函数构建了文档词集模型，只包含了文档出现与否
    """
    return_vec = [0]*len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] = 1
        else:
            print('the word: {} is not in my vocabulary!'.format(word))
    return return_vec

def bag_of_word2vec(vocab_list, input_set):
    """
    构建文档词袋模型，记录文中出现的单词的次数
    """
    return_vec = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] +=1
    return return_vec

def trainNB0(train_matrix, train_category):
    """
    train_matrix: list, 文档矩阵，train_matrix[i]是第i个文档
    train_category: list, 文档的类别向量，train_category[i]第i个文档的类别
                    类别只有0、1 不是/是侮辱性文档
    这个函数用来训练一个朴素贝叶斯分类器，实质上就是统计训练样本，计算各种概率
    """
    #print(len(train_matrix))
    #print(len(train_category))
    assert len(train_matrix) == len(train_category)
    # 训练文档的数目
    num_train_docs = len(train_matrix)
    # 每个训练文档中含有的单词数
    num_wrods = len(train_matrix[0])
    # 侮辱性文档占比（概率）
    p_abusive = sum(train_category)/float(num_train_docs)
    # [单词i出现时这个文档不属于侮辱性文档的次数 for 单词i in 单词列表]
    p0_num = np.ones(num_wrods)
    # [单词i出现时这个文档属于侮辱性文档的次数 for 单词i in 单词列表]
    p1_num = np.ones(num_wrods)
    # 侮辱性文档的个数，以及不是侮辱性文档的个数
    # 为了防止某个单词对的 概率是0相乘概率就变成0的情况，把这个基数变为2
    p0_denom = 2.0
    p1_denom = 2.0
    for i in range(num_train_docs):
        if train_category[i] == 1:
            # train_matrix[i]是一个文档向量，由1/0组成，长度为num_words
            p1_num += train_matrix[i]
            p1_denom += 1 # 此处勘误，原书中是sum(trainMatrix[i])
        else:
            p0_num += train_matrix[i]
            p0_denom += 1 # 此处勘误，原书中是sum(trainMatrix[i])
    # p1_vec = [p(wi=1|ci=1) for wi in 单词表]
    # log是为了防止概率太小而下溢
    p1_vec = np.log(p1_num/p1_denom)
    p0_vec = np.log(p0_num/p0_denom)
    return p0_vec, p1_vec, p_abusive

def classifyNB(vec2classify, p0_vec, p1_vec, p_class1):
    """
    vec2classify: list,需要被分类的向量
                其长度需要和单词表一样长，且单词的排列顺序需要和单词表一样
    p0_vec: list, [p(wi=1|ci=0) for wi in 单词表]
    p1_vec: list, [p(wi=1|ci=1) for wi in 单词表]
    p_class1: float, 文档输出类别1，侮辱性文档的概率
    """
    # vec2classify 中的值为1或者0，所以和p1_vec相乘之后再求和
    # 就得到log(p(w1,w2,...,wn|ci=1)/p(w1,w2,...,wn)) 
    # 再加上log(p_class1)就相当于除以log(p_class0)
    # 那么就得到了log(p(ci=1)*p(w1,w2,...,wn|c)/p(w1,w2,...,wn)) = log(p(ci=1|w1,w2,...,wn))
    p1 = sum(vec2classify * p1_vec) + np.log(p_class1)
    p0 = sum(vec2classify * p0_vec) + np.log(1.0 - p_class1)
    return 1 if p1 > p0 else 0

def test_NB():
    list_posts, list_classes = load_dataset()
    my_vocab_list = create_vocab_list(list_posts)
    train_mat = []
    for post_in_doc in list_posts:
        train_mat.append(set_of_words2vec(my_vocab_list, post_in_doc))
    p0_v, p1_v, p_ab = trainNB0(train_mat, list_classes)
    test_entry = ['love', 'my', 'dalmation']
    this_doc = np.array(set_of_words2vec(my_vocab_list, test_entry))
    print(test_entry, 'classified as: ', classifyNB(this_doc, p0_v, p1_v, p_ab))
    test_entry = ['stupid', 'garbage']
    this_doc = np.array(set_of_words2vec(my_vocab_list, test_entry))
    print(test_entry, 'classified as: ', classifyNB(this_doc, p0_v, p1_v, p_ab))

if __name__ == '__main__':
    test_NB()
