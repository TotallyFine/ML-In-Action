# coding:utf-8
import numpy as np

def tanh(x):
    """
    x: float/int/ndarray
    这个函数返回tanh(x)的结果
    """
    return np.tanh(x)

def tanh_deriv(x):
    """
    x: float/int/ndarray
    这个函数用于计算tanh在某一点的导数
    """
    return 1.0 - np.tanh(x) * np.tanh(x)

def logistic(x):
    """
    x: float/int/ndarray
    这个函数用于计算logistic函数的结果
    """
    return 1 / (1 + np.exp(-x))

def logistic_derivative(x):
    """
    x: float/int/ndarray
    求logistic函数在某一点的导数
    """
    return logistic(x) * (1 - logistic(x))

class NeuralNetwrok:
    def __init__(self, layers, activation='tanh'):
        """
        layers: list, 一维，定义了每层的神经元数，第一层的神经元要和特征数相同
        activation: str, 激活函数
        """
        # 不同的激活函数使用不同的导函数
        if activation == 'logistic':
            self.activation = logistic
            self.activation_deriv = logisitic_deriv
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv

        self.weights = []
        # 对权重进行初始化，每个多出来的1都是bias
        # 我参考的那个链接在初始化上似乎并不对，下面是原始的初始化方法
        # 我使用的是我更正后的初始化方法
        #for i in range(1, len(layers) - 1):
        #    self.weights.append((2 * np.random.random((layers[i-1]+1,layers[i]+1))-1)*0.25)
        #    self.weights.append((2 * np.random.random((layers[-2]+1,layers[-1]))-1)*0.25)
        for i in range(len(layers)):
            if i == 0:
                self.weights.append((2 * np.random.random((layers[i]+1,layers[i]+1))-1)*0.25)
            else:
                self.weights.append((2 * np.random.random((layers[i-1]+1,layers[i]+1))-1)*0.25)

    def fit(self, X, y, learning_rate=0.2, epochs=10000):
        """
        X: list/ndarray, 训练数据，二维，X[i, j]第i条数据的第j个属性
        y: list/ndarray, 标签，一维，y[i]第i个数据的标注
        learning_rate: double, 学习率
        epochs: int, 训练的轮次数
        """
        # 将数据规整成为2维，只有一个标称数值或者一维的向量的时候在外面再套一维成为二维
        X = np.atleast_2d(X)
        # 多出来的一维是bias初始值值为1，X.shape[0]是数据个数，X.shape[1]是特征个数
        temp = np.ones([X.shape[0], X.shape[1]+1])
        # 将数据放入temp中，最后一个（-1）空出来，仍然是1这个给bias留得
        temp[:, 0:-1] = X
        X = temp
        y = np.array(y)

        for k in range(epochs):
            # 随机选择一条数据
            i = np.random.randint(X.shape[0])
            # a是二维的，第i条数据
            a = [X[i]]
            #print(a)
            #print(self.weights)
            # print(self.weights.shape)
            # l意为layer，对于每个层计算数据和权重的点积再使用激活函数进行计算
            # 每层计算出来的结果直接加到a的最后，a[0]是数据，a[1]就是第一层的输出结果然后和weights[1]计算第二层的输出结果
            #print('forward')
            for l in range(len(self.weights)):
                #print(l)
                #print(a[l].shape)
                #print(self.weights[l].shape)
                a.append(self.activation(np.dot(a[l], self.weights[l])))
            #print('in a')
            #for j in a:
            #    print(j.shape)
            # 用最后一层的输出和标签进行做差计算error
            error = y[i] - a[-1]
            # 梯度下降
            deltas = [error * self.activation_deriv(a[-1])]
            # 开始反向传播，从倒数第二层开始计算，因为倒数第一层已经计算导数
            # 利用了同样的技巧，将每次计算得到的deltas附加在数组的最后
            # deltas[0]是最后一层的输出的delta，用deltas[0]来计算deltas[1],deltas[1]就是倒数第二层的delta
            for l in range(len(a) - 2, 0, -1):
                deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_deriv(a[l]))
            # 将数组翻转，得到正常顺序的delta
            deltas.reverse()
            # 进行梯度下降
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)

    def predict(self, x):
        """
        x: list, 测试数据
        对测试数据做出预测
        """
        x = np.array(x)
        # 多出来的1是bias
        temp = np.ones(x.shape[0]+1)
        temp[0:-1] = x
        a = temp
        # 前向传播，先乘权重再激活
        # 每循环一次，a就变成这层的输出，最后就是最终层的输出
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a

def test():
    nn = NeuralNetwrok([2, 2, 3, 1], 'tanh')
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])
    nn.fit(X, y)
    for i in [[0, 0], [0, 1], [1, 0], [1, 1]]:
        print(i, nn.predict(i))

if __name__ == '__main__':
    test()