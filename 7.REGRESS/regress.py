# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    """
    这个函数用于读取数据，存放在文件中的数据必须是一行一条数据
    每个特征用\t隔开，每行最后是标注
    """
    numFeat = len(open(fileName).readline().split('\t')) - 1
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

def standRegress(xArr, yArr):
    """
    xMat: list, dataMat
    yMat: list, labelMat
    return: ws: ndarray误差最小的权重系数
    此函数用于标准回归，但是只适用于xTx可逆时
    """
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    xTx = xMat.T * xMat
    if np.linalg.det(xTx) == 0.0:
        print('This matrix is singular, cannot do inverse')
        return
    ws = xTx.I * (xMat.T*yMat)
    return ws

def testStandRegress():
    xArr, yArr = loadDataSet('ex0.txt')
    ws = standRegress(xArr, yArr)
    print(ws)
    xMat = np.mat(xArr)
    yMat = np.mat(yArr)
    yHat = xMat*ws
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])
    xCopy = xMat.copy()
    # 将点按照升序排序
    xCopy.sort(0)
    yHat = xCopy*ws
    ax.plot(xCopy[:, 1], yHat)
    plt.show()

def lwlr(testPoint, xArr, yArr, k=1.0):
    """
    testPoint: tuple/list, 用于测试的点
    xArr: list
    yArr: list
    k: double, 平滑参数，这个值越小则拟合的线越扭曲，越和点贴近
               但是这个值太小会导致过拟合，太大会欠拟合
    局部加权线性回归
    """
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    m = np.shape(xMat)[0]
    weights = np.mat(np.eye((m)))
    for j in range(m):
        diffMat = testPoint - xMat[j, :]
        weights[j, j] = np.exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx = xMat.T * (weights * xMat)
    if np.linalg.det(xTx) == 0.0:
        print('This matrix is singular, cannot do inverse')
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint*ws

def lwlrTest(testArr, xArr, yArr, k=1.0):
    """
    testArr: ndarray, 测试集
    xArr: list
    yArr: list
    k: double, 平滑系数
    这个函数用于对测试集做出预测
    """
    m = np.shape(testArr)[0]
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat 

def rssError(yArr, yHatArr):
    return ((yArr-yHatArr)**2).sum()

def abalone():
    """
    预测鲍鱼年龄
    """
    abX, abY = loadDataSet('abalone.txt')
    yHat01 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 0.1)
    yHat1 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 1)
    yHat10 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 10)
    print('k=0.1 error is', rssError(abY[0:99], yHat01.T))
    print('k=1 error is', rssError(abY[0:99], yHat1.T))
    print('k=10 error is', rssError(abY[0:99], yHat10.T))
  
def ridgeRegress(xMat, yMat, lam=0.2):
    """
    xMat: list, 
    yMat: list,
    la: double, lambda 缩减系数
    这个函数进行岭回归，来得到权重
    """
    xTx = xMat.T * xMat
    # 缩减系数乘以单位矩阵再加到xTx上
    denom = xTx + np.eye(np.shape(xMat)[1])*lam
    # 仍需判断矩阵是否奇异
    if np.linalg.det(denom) == 0.0:
        print('This matrix is singular, cannot do inverse')
        return
    # .I是求逆矩阵， .A是转换成ndarray
    ws = denom.I * (xMat.T*yMat)
    return ws

def ridgeTest(xArr, yArr):
    """
    xArr: list, 
    yArr: list, 
    return: wMat: list, 包含使用不同的缩减系数训练出来的多个权重系数
    """
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    yMean = np.mean(yMat, 0)
    yMat = yMat - yMean
    # 得到方差
    xMeans = np.mean(xMat, 0)
    xVar = np.var(xMat, 0)
    # 进行归一化/标准化
    xMat = (xMat - xMeans)/xVar
    numTestPts = 30
    # 初始化权重系数
    wMat = np.zeros((numTestPts, np.shape(xMat)[1]))
    # 使用不同的权重进行多次回归，并记录
    for i in range(numTestPts):
        ws = ridgeRegress(xMat, yMat, np.exp(i-10))
        wMat[i, :]=ws.T
    return wMat

def ridgePlot():
    """
    这个函数对鲍鱼年龄进行训练并将训练好的权重结果显示出来
    图的最左边是缩减系数为0，得到的权重结果和线性回归一样
    最右边是设置中的缩减系数最大的情况，系数全部缩减为0
    在中间的某个部分就可以得到最好的系数，还需要交叉验证
    另外可以通过系数的绝对值的大小来判断哪个权重有影响力，越大影响力越大
    """
    abX, abY = loadDataSet('abalone.txt')    
    ridgeWeights = ridgeTest(abX, abY)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ridgeWeights)
    plt.show()

def regularize(xMat):
    """
    xMat: np.matrix，每一行是一条数据，每一列是一类属性
    这个函数对输入的矩阵按照列进行规范化
    """
    inMat = xMat.copy()
    # 求均值方差，然后进行标准化
    inMeans = np.means(inMat, 0)
    inVar = np.var(inMat, 0)
    inMat = (inMat - inMenas)/inVar
    return inMat

def stageWise(xArr, yArr, eps=0.01, numIt=100):
    """
    xArr: list
    yArr: list
    eps: double, 每次迭代的步长
    numIt: int, 迭代次数
    return: returnMat: matrix 二维的，记录了每次贪心之后的系数
    这个函数实现逐步线性回归
    """
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    yMean = np.mean(yMat, 0)
    yMat = yMat - yMean
    # 标准化
    xMat = regularize(xMat)
    m, n = np.shape(xMat)
    returnMat = np.zeros((numIt, n))
    # 初始化权重系数
    ws = np.zeros((n, 1))
    wsTest = ws.copy()
    wsMax = ws.copy()
    # 开始迭代
    for i in range(numIt):
        print(ws.T)
        lowestError = float('inf')
        # 遍历每个属性，对每个属性都尝试增加减小,选取最好的情况
        for j in range(n):
            for sign in [-1, 1]:
                wsTest = ws.copy()
                wsTest[j] += eps*sign
                wsTest = xMat*wsTest
                # 计算误差 看是否变小
                rssE = rssError(yMat.A, yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        # 记录本次最好贪心结果
        ws = wsMax.copy()
        returnMat[i, :] = ws.T
    return returnMat

def 

if __name__ == '__main__':
    # testStandRegress()
    ridgePlot()