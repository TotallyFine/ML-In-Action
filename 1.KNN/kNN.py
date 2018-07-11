# coding:utf-8
import numpy as np
import operator # 运算符模块

def create_dataset():
    group = np.array([[1.0,1.1], [1.0,1.0], [0,0], [0,0.3]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0(inX, dataset, labels, k):
    """
    inX: ndarray, test data
    dataset: 2 dimension numpy array, dataset[i] contain i'th data
             dataset[i][0] 0'th feature
    labels: numpy array, labels[i] i'th data's label
    k: int, num of nearset neighbour
    使用欧式距离计算
    """
    assert isinstance(dataset, np.ndarray)
    assert len(inX) == len(dataset[0])
    #assert isinstance(labels, list)
    dataset_size = dataset.shape[0]
    # 先扩充数据至和dataset一样大，然后相减
    diff_mat = np.tile(inX, (dataset_size, 1)) - dataset
    sq_diff_mat = diff_mat ** 2 # 平方
    # 每个样本的不同属性的差值相加
    # len(sq_distamces) == dataset_size
    sq_distances = sq_diff_mat.sum(axis=1) 
    distances = sq_distances**0.5
    # 获得排序下标
    sorted_distance_indicies = distances.argsort()
    class_count = {}
    # 选择距离最小的k个样本
    for i in range(k):
        # 第i个最接近的标签
        vote_i_label = labels[sorted_distance_indicies[i]]
        # 为这个标签的“分数”加一，如果这个标签还没有存在与class count中那么就返回0
        class_count[vote_i_label] = class_count.get(vote_i_label, 0) + 1
        # 根据标签已经得到的分数进行排序，一般的排序是从小到达，这里翻转一下，所以下面取0
        sorted_class_count = sorted(class_count.items(), 
            key=operator.itemgetter(1), reverse=True)
    # print(sorted_class_count)
    # [('A', 2), ('B', 2)] operator.itemgetter(1) 取出分数进行排序
    return sorted_class_count[0][0]

def file2matrix(filename, num_features=3):
    """
    num_features: int, Features numbers
    从文件中读取数据，每条数据一行，每个特征之间用\t分割。最后一个是标签
    """
    fr = open(filename)
    lines = fr.readlines()
    num_of_lines = len(lines)
    return_mat = np.zeros((num_of_lines, num_features))
    class_label = []
    index = 0
    for line in lines:
        line = line.strip()
        list_format_line = line.split('\t')
        return_mat[index, :] = list_format_line[0:num_features]
        class_label.append(int(list_format_line[-1])) # 最后一个是标签
        index += 1
    return return_mat, class_label

def auto_norm(dataset):
    """
    对数据进行归一化
    dataset: ndarray, 二维的数组
    """
    assert isinstance(dataset, np.ndarray)
    min_vals = dataset.min(0)
    max_vals = dataset.max(0)
    ranges = max_vals - min_vals
    norm_dataset = np.zeros(dataset.shape)
    m = dataset.shape[0]
    # np.tile(array, (line, 1)) 扩展成最外层1个，里面m个array
    norm_dataset = dataset - np.tile(min_vals, (m, 1))
    norm_dataset = norm_dataset/np.tile(ranges, (m, 1))
    return norm_dataset, ranges, min_vals

def main():
    dataset, labels = create_dataset()
    print(classify0(np.array([1.1, 1.2]), dataset, labels, 4))

if __name__ == '__main__':
    #print(file2matrix('data.txt', 2))
    print(auto_norm(np.array([[1,4,6], [3, 8, 0], [3, 2, 6]])))