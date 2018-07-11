# coding:utf-8
# 决策树的计算负责度不高，输出结果易于理解，对中间值的缺失不敏感
# 可以处理不相关特征数据
# 在进行学习之前需要对数据特征进行评估，找到当前数据集上哪个特征
# 对分类起到决定性的作用，在完成评估之后数据就会被划分成几个子集
# 这些子集会分布在第一个决策点的所有分支上，如果某个分支下的数据
# 属于统一类型，则已经正确分类，如果不属于统一类型，则需要再进行
# 划分，知道所有具有相同类型的数据均在一个数据子集中
#
# 连续的数据需要被离散化，标签类的数据需要数字化
# 
# 这里使用ID3算法 
# import numpy as np 没有使用到numpy，全是纯python
from math import log
import operator

from plot import retrieve_tree, create_plot

def majority_cnt(class_list):
    """
    class_list: list, 数据的标签，这个函数用于选出数据中最多的标签
    当数据中只剩下一个特征的时候没办法再进行分支，所以用训练的数据
    中最多的类来决定最后的分类
    """
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1
    sorted_class_count = sorted(class_count.items(),
        key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]

def calc_shannon_ent(labels):
    """
    labels: list, 标签，一维数组
    计算标签中的香农熵,除了熵之外另一个度量集合无序程度的方法是基尼不纯度
    """
    num_entries = len(labels)
    label_counts = {}
    for label in labels:
        if label not in label_counts.keys():
            label_counts[label] = 0
        label_counts[label] += 1
    shannon_ent = 0.0
    for key in label_counts:
        # 先这个标签出现的概率
        prob = float(label_counts[key])/num_entries
        # 熵 = Σ-(probi)*log2(probi)
        shannon_ent -= prob * log(prob, 2)
    return shannon_ent

def split_dataset(dataset, labels, axis, value):
    """
    dataset: list, 数据集
    axis: int, 按照某个特征的某个阈值来划分数据集
    value: float, 划分特征的阈值
    划分完数据集之后就要进行度量熵是否变小了
    """
    ret_dataset = []
    ret_labels = []
    for i, feat_vec in enumerate(dataset):
        # 一旦发现选中的特征等于这个阈值就进行分割
        # 符合这个阈值的记录重新组成一个新的数据集，并将这个特征去掉
        if feat_vec[axis] == value:
            reduced_feat_vec = feat_vec[:axis]
            reduced_feat_vec.extend(feat_vec[axis+1:])
            ret_dataset.append(reduced_feat_vec)
            ret_labels.append(labels[i])
    return ret_dataset, ret_labels

def choose_best_feature_to_split(dataset, labels):
    """
    dataset: list, 只包含特征的数据集，没有标签
    返回最适合用于分类的那个特征，以及
    """
    #assert isinstance(dataset, np.ndarray)
    assert len(dataset) == len(labels)
    num_features = len(dataset[0])
    # 计算原始数据集的熵
    base_entropy = calc_shannon_ent(labels)
    best_info_gain = 0.0
    best_feature = -1
    # 对每个特征来说，尝试每一个存在的值作为阈值
    for i in range(num_features):
        # 所有条数据中这个特征的所有取值
        feat_list = [example[i] for example in dataset]
        # 集合，使得没有重复
        unique_vals = set(feat_list)
        new_entropy = 0.0
        for value in unique_vals:
            # 划分数据集
            sub_dataset, sub_labels = split_dataset(dataset, labels, i, value)
            # 计算熵
            prob = len(sub_dataset)/float(len(dataset))
            new_entropy += prob * calc_shannon_ent(sub_labels)
        info_gain = base_entropy - new_entropy
        # 每种阈值划分之后的熵是否比原始数据集的熵更低
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature

def create_tree(dataset, labels, feature_name):
    """
    dataset: list, 数据集
    labels: list, 标签,dataset和labels的长度相同，一条数据对应一个label
    feature_name: list, 每个feature对应的实际名称，用于构造字典

    递归构造决策树，递归结束的条件是，程序遍历完所有划分数据集的属性
    或者每个分支下的所有实例都具有相同的分类。如果所有实例都具有相同的
    分类，则得到一个叶子节点或者终止块，任何达到叶子接待你的数据必然属于
    叶子节点的分类
    """
    assert len(dataset[0]) == len(feature_name)
    class_list = labels
    # 如果剩余的数据全部属于同一个类别就停止划分
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    # 只剩一个特征没有被划分就停止划分
    if len(dataset[0]) == 1:
        # 此时返回剩余训练数据中最多的类别作为终止节点
        return majority_cnt(class_list)
    best_feature = choose_best_feature_to_split(dataset, labels)
    # 得到这个用于划分的特征的名称
    best_feature_name = feature_name[best_feature]
    # 构造树
    my_tree = {best_feature_name:{}}
    del(feature_name[best_feature]) # 从中删除，这个特征已经放到了树上
    # 得到这个特征的所有取值
    feat_values = [example[best_feature] for example in dataset]
    # 得到这个特征的取值范围，用于选出最好的阈值作为划分标准
    unique_vals = set(feat_values)
    # 尝试每个值，有多少个值就有多少个分支
    for value in unique_vals:
        # 使得每次不会改变原来的feature_name
        sub_feature_name = feature_name[:]
        # 划分数据集，作为一个分支
        sub_dataset, sub_labels=split_dataset(dataset, labels, best_feature, value)
        my_tree[best_feature_name][value] = create_tree(sub_dataset, sub_labels, sub_feature_name)
    return my_tree

def classify(input_tree, feature_name, test_vec):
    """
    input_tree: dict, 算法构造的树
    feature_name: list, 特征的名字
    test_vec: list, 等待分类的特征向量
    """
    assert isinstance(input_tree, dict)
    assert len(test_vec) == len(feature_name)
    # 得到第一个用于分类的特征的名称
    first_str = list(input_tree.keys())[0]
    second_dict = input_tree[first_str]
    feat_index = feature_name.index(first_str)
    # 遍历这个特征的所有取值，如果test_vec中的这个特征的取值是树中的这个取值就进入这个分支
    # 如果是这个特征的某个取值，但是已经到了最终的分分类节点就算完成了分类
    for key in second_dict.keys():
        if test_vec[feat_index] == key:
            if type(second_dict[key]).__name__ == 'dict':
                class_label = classify(second_dict[key], feature_name, test_vec)
            else:
                class_label = second_dict[key]
    return class_label

def save_tree(input_tree, filename):
    """
    input_tree: dict, 算法生成的决策树
    filename: str,保存的文件名
    这个函数用来保存已经生成的决策树
    """
    import pickle
    fw = open(filename, 'wb')
    # 第三个0代表使用ASCII协议，不会导致无法打开的问题
    pickle.dump(input_tree, fw, 0)
    fw.close()

def load_tree(filename):
    """
    filename: str, 从文件中读取已经生成的决策树
    """
    import pickle
    fr = open(filename, 'rb')
    #print(fr.readlines())
    return pickle.load(fr)

def get_lenses(filename='lenses.txt'):
    fr = open(filename)
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    feature_name = ['age', 'prescript', 'astigmatic', 'tear_rate']
    assert len(lenses[0]) == 5
    labels = [inst[4] for inst in lenses]
    lenses = [inst[:4] for inst in lenses]
    return lenses, labels, feature_name

if __name__ == '__main__':
    #feature_name = ['no surfacing', 'flippers']
    #my_tree = retrieve_tree(0)
    #save_tree(my_tree, 'classify_tree.pkl')
    #print(load_tree('classify_tree.pkl'))
    #print(classify(my_tree, feature_name, [1, 1]))
    
    lenses, labels, feature_name = get_lenses()
    lenses_tree = create_tree(lenses, labels, feature_name)
    create_plot(lenses_tree)