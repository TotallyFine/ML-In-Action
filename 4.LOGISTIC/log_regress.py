# coding:utf-8
# 梯度上升优化算法进行回归
# 特征向量X=[x1, x2, ..., xn]
# z = W.*X = w1*x1 + w2*x2 + ...+ wn*xn
# y = 1/(1+e^(-z))即为预测出来的标签
# 然后通过损失函数得到error，使用error进行反向传播
# 为了优化系数，使用梯度下降，求每个系数的偏微分，得到梯度
# Δw = ∂error/∂w
# 再根据梯度和学习率更新系数
# w = w - α*Δw
# 在LOGISTIC回归中使用对数似然损失函数
# 根据数学原理简化后系数的更新可以直接写成
# weights = weights + α*x.T*(label-y)
# 具体的简化过程见
# http://secfree.github.io/blog/2017/01/01/questions-about-logistic-regression-in-machine-learning-in-action-and-full-explanation.html

import numpy as np
import random

def load_dataset():
    """
    从文本文件中读取数据，每行只有3个数据，x1 x2 label 都是数字，用空格分割
    """
    data_mat = []
    label_mat = []
    fr = open('test_set.txt')
    for line in fr.readlines():
        line_arr = line.strip().split()
        data_mat.append([1.0, float(line_arr[0]), float(line_arr[1])])
        label_mat.append(int(line_arr[2]))
    return data_mat, label_mat

def sigmoid(in_x):
    """
    返回sigmoid函数的结果
    函数对in_x求导结果为(1-sigmoid(in_x))*sigmoid(in_x)
    """
    #assert isinstance(in_x, np.ndarray)
    return 1.0/(1+np.exp(-in_x))

def stoc_grad_ascent0(data_matrix, class_labels, alpha=0.01, weights=None):
    """
    data_mat: list, 二维数组，data_mat[i]第i个数据，data_mat[i, 1]第i个数据的第2个特征
    class_labels: list, 一维数组class_labels[i]是第i个数据的label
    alpha: float, 学习率
    随机梯度下降法，一次只有一条数据被利用，有多少条数据就迭代多少次
    """
    m, n = np.shape(data_matrix)
    if weights is None:
        weights = np.ones(n)
    for i in range(m):
        h = sigmoid(sum(data_matrix[i]*weights))
        error = class_labels[i] - h
        weights = weights + alpha * error * data_matrix[i]
    return weights

def stoc_grad_ascent1(data_matrix, class_labels, num_iter=50, weights=None):
    """
    data_mat: ndarray, 二维数组，data_mat[i]第i个数据，data_mat[i, 1]第i个数据的第2个特征
    class_labels: list, 一维数组class_labels[i]是第i个数据的label
    num_iter: int, 迭代的次数
    改进版的随机梯度下降，每次更新weights前学习率都要更新
    在这个版本中的随机梯度下降中，系数不会随着迭代而周期性的波动
    并且随着迭代次数的增大，每次震荡波动的幅度越来越小
    这个版本收敛更快
    """
    m, n = np.shape(data_matrix)
    if weights is None:
        weights = np.ones(n)
    # 迭代num_iter轮
    for j in range(num_iter):
        # 每次随机从data_index中得到一个下标，作为这次用于训练的数据
        # 使用过这个数据后就删除这个下标，在这轮迭代就不再使用
        data_index = list(range(m))
        for i in range(m):
            # 学习率会随着迭代次数不断减小，但永远不会减小到0
            alpha = 4/(1.0+j+i)+0.01
            rand_index = int(random.uniform(0, len(data_index)))
            h = sigmoid(sum(data_matrix[rand_index]*weights))
            error = class_labels[rand_index] - h
            weights = weights + alpha * error * data_matrix[rand_index]
            del(data_index[rand_index])
    return weights


def grad_ascent(data_mat, class_labels, alpha=0.001, max_cycles=500, weights=None):
    """
    data_mat: list, 二维数组，data_mat[i]第i个数据，data_mat[i, 1]第i个数据的第2个特征
    class_labels: list, 一维数组class_labels[i]是第i个数据的label
    alpha: float, 学习率
    max_cycles: int, 最大epoch
    weights: ndarray, 权重，可以把预训练的权重放进去
    返回的weights是ndarray类型，训练好的系数
    这里名字是梯度上升，实则是将loss函数取负，没什么区别
    """
    # 这里使用numpy中的matrix类型，matrix是一种特殊的ndarray
    # 其维度不会随着运算而改变，一直为2
    data_matrix = np.mat(data_mat)
    # 将label矩阵进行转置，使得能够利用矩阵相乘直接进行计算
    label_mat = np.mat(class_labels).transpose()
    # m条数据，n个特征
    m, n = data_matrix.shape
    if weights is None:
        # n个特征，n行
        weights = np.ones((n, 1))
    for k in range(max_cycles):
        # 下面这条代码为矩阵相乘，不是按batch，而是每次直接全部进行计算
        h = sigmoid(data_matrix*weights)
        # 得到error
        error = (label_mat - h)
        weights = weights + alpha * data_matrix.transpose() * error
    return weights

def plot_best_fit(wei):
    """
    wei: np.matrix,最佳系数
    此函数根据训练结果绘制图像
    """
    import matplotlib.pyplot as plt
    # 从np.matrix中得到ndarray
    if isinstance(wei, np.matrix):
        weights = wei.getA()
    if isinstance(wei, np.ndarray):
        weights = wei
    # 加载数据
    data_mat, label_mat = load_dataset()
    data_arr = np.array(data_mat)
    # n条数据
    n = np.shape(data_arr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    # 遍历所有数据分别加到两个集合中去
    for i in range(n):
        if int(label_mat[i]) == 1:
            xcord1.append(data_arr[i, 1])
            ycord1.append(data_arr[i, 2])
        else:
            xcord2.append(data_arr[i, 1])
            ycord2.append(data_arr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # 散点图做出所有数据
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    # 作出训练好的拟合直线
    x= np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

def classify_vector(in_x, weights):
    prob = sigmoid(sum(in_x*weights))
    return 1 if prob > 0.5 else 0

if __name__ == '__main__':
    data_arr, label_mat = load_dataset()
    weights = stoc_grad_ascent1(np.array(data_arr), label_mat)
    plot_best_fit(weights)