# coding:utf-8
import numpy as np

def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = [i for i in map(float, curLine)]
        dataMat.append(fltLine)
    return dataMat

def binSplitDataSet(dataSet, feature, value):
    """
    dataSet: np.matrix, 需要被分割的数据，二维
    feature: int, 按照哪个特征来被分割
    value: double, 被分割的阈值
    这个函数实现二元分割法，将数据分成两份
    """
    # 先使用nonzero来筛选>value的那些数据，形成索引
    # 下面两行，《机器学习实战》书中162页处错了
    # 这两行的最后都没有[0]
    mat0 = dataSet[np.nonzero(dataSet[:, feature] > value)[0], :]
    mat1 = dataSet[np.nonzero(dataSet[:, feature] <= value)[0], :]
    return mat0, mat1

def regLeaf(dataSet):
    """
    这个函数用于构建叶节点，每个叶节点实际上就是一个值
    被划分到这个节点的所有数据的均值
    """
    return np.mean(dataSet[:, -1])

def regErr(dataSet):
    """
    计算总方差
    """
    return np.var(dataSet[:, -1]) * np.shape(dataSet)[0]

def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    """
    dataSet: ndarray
    leafType: function, 建立叶结点的函数
    errType: function 计算误差的函数
    ops: tuple 其他的参数
    这个函数实现递归构建树，每个树都是字典的形式
    {'spInd':splitIndex, 'spVal': splitValue, 'left':leftTree also dict,
    'right':rightTree also dict}
    """
    # 选择最好的划分的特征和阈值
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat == None:
        return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    # 递归构建树
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree

def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    """
    dataSet: ndarray,
    leafType: function, 生成叶结点的函数
    errType: function, 计算混乱度
    ops: tuple, 生成叶结点的其他选项 ops[0]允许的最小下降的混乱度
         ops[1] 允许每次二分数据集后最小的数据集中拥有的数据个数
    这个函数遍历所有的数据，从数据中选择能使混乱度最低的特征和阈值
    """
    # tolS是允许误差下降的最小值
    # tolN是允许的的切分后的最小样本数
    tolS = ops[0]
    tolN = ops[1]
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:
        return None, leafType(dataSet)
    m, n = np.shape(dataSet)
    # 计算数据的混乱度
    S = errType(dataSet)
    bestS = float('inf')
    bestIndex = 0
    bestValue = 0
    dataSet = np.array(dataSet)
    # 遍历每个特征，从中选择特征和阈值使得混乱度最小
    for featIndex in range(n-1):
        # 使用set可以使值唯一      
        for splitVal in set(dataSet[:, featIndex]):
            # 进行二元分割
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            #print(mat0.shape)
            #print(mat1.shape)
            # 如果分割后的两个数据集中有一个已经小道无法再次划分则换下一个值
            if np.shape(mat0)[0] <= tolN or np.shape(mat1)[0] <= tolN:
                continue
            # 计算新的混乱度
            newS = errType(mat0) + errType(mat1)
            # 如果混乱度更小则更新
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    # 误差下降的太小了，不允许再划分了
    if S - bestS < tolS:
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if np.shape(mat0)[0] <= tolN or np.shape(mat1)[0] <= tolN:
        return None, leafType(dataSet)
    return bestIndex, bestValue

def testTree():
    myDat = loadDataSet('ex00.txt')
    myMat = np.mat(myDat)
    tree = createTree(myMat)
    print(tree)

def isTree(obj):
    """
    用于判断是不是树，在后剪枝中用到
    """
    return type(obj).__name__ == 'dict'

def getMean(tree):
    """
    tree: dict, 通过CART算法构建好的树
    这个函数从上到下遍历树直到叶子结点。如果找到叶子结点则计算平均值。
    该函数对树进行塌陷处理（即返回树平均值）
    """
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left'] + tree['right'])/2

def prune(tree, testData):
    """
    tree: dict, 用CART构建好的树
    testData: ndarray, 测试集
    这个函数返回剪枝后的树
    """
    if np.shape(testData)[0] == 0:
        return getMean(tree)
    if isTree(tree['right']) or isTree(tree['left']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    # 递归实现剪枝
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'], rSet)
    # 递归到最底端，树的两个分叉都是叶子节点，尝试将这两个叶子节点合并
    # 看是否能降低混乱度
    if not isTree(tree['left']) and not isTree(tree['right']):
        # 先通过这两个叶子节点的树根上的index和value二分数据
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        # 计算不合并时的混乱度，tree['right']是一个叶子节点，保存的就是一个值
        # 混乱度就是总方差，因为是用tree['left']来代表lSet所以用lSet中的所有数据减去tree['left']
        errorNoMerge = np.sum(np.power(lSet[:, -1] - tree['left'], 2)) + np.sum(np.power(rSet[:, -1] - tree['right'], 2))
        # 计算合并以后的混乱度，合并以后就是用tree['right'] tree['left']的均值来代表
        treeMean = (tree['right'] + tree['left']) / 2
        errorMerge = np.sum(np.power(testData[:, -1] - treeMean, 2))
        # 如果合并以后更小就用合并后的那个值来代替这个树的两个节点，使得这个子树变成一个节点
        if errorMerge < errorNoMerge:
            print('merging')
            return treeMean
        else:
            return tree
    else:
        return tree

def testPrune():
    """
    这个函数用于测试剪枝
    """
    myDat2 = loadDataSet('ex2.txt')
    print(len(myDat2[0]))
    myMat2 = np.mat(myDat2)
    myTree = createTree(myMat2, ops=(0, 1))
    myDatTest = loadDataSet('ex2test.txt')
    myMat2Test = np.mat(myDatTest)
    prunedTree = prune(myTree, myMat2Test)
    print(prunedTree)

def linearSolve(dataSet):
    """
    这个函数用于对分割后的子数据集进行线性拟合
    """
    m, n = np.shape(dataSet)
    X = np.mat(np.ones((m, n)))
    Y = np.mat(np.ones((m, 1)))
    X[:, 1:n] = dataSet[:, 0:n-1]
    # 下面Y的shape可能为(m,)
    Y = dataSet[:, -1]
    # 将Y变为(m, 1)
    if len(Y.shape) == 1:
        Y = Y[:, np.newaxis]
    xTx = X.T*X
    if np.linalg.det(xTx) == 0:
        raise NameError('This matrix is singular, cannot do inverse, \n try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws, X, Y

def modelLeaf(dataSet):
    """
    利用CART树进行回归的时候这个函数用于生成叶节点
    """
    ws, X, Y = linearSolve(dataSet)
    return ws

def modelErr(dataSet):
    """
    计算线性拟合的error，总方差
    """
    ws, X, Y = linearSolve(dataSet)
    yHat = X * ws
    return np.sum(np.power(Y - yHat, 2))

def testLinearTree():
    """
    这个函数用于测试利用树实现分段线性
    """
    myMat2 = np.mat(loadDataSet('exp2.txt'))
    tree = createTree(myMat2, modelLeaf, modelErr, (1, 10))
    print(tree)

if __name__ == '__main__':
    testLinearTree()