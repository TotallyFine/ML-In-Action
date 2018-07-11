# coding:utf-8
import numpy as np

def loadSimpData():
    """
    这个函数用于产生数据
    """
    datMat = np.matrix([[1., 2.1],
        [2., 1.1],
        [1.3, 1.],
        [1., 1.],
        [2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat, classLabels

def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    """
    dataMatrix: np.matrix, 需要被分类的数据
    dimen: int, 以哪个属性为依据进行分类
    threshVal: double, 分类的阈值
    threshIneq: str, 当为'lt'的时候小于阈值的被分为-1类，大于的被分为+1类
                当不为'lt'的时候，相反。
    return: retArray: 返回的分类结果。
    这个函数通过阈值进行分类，分为-1或者+1.
    """
    retArray = np.ones((np.shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray

def buildStump(dataArr, classLabels, D):
    """
    dataArr: np.matrix, 训练数据
    classLabels: list, 标签
    D: 每个数据的权重
    这个函数用于构造一系列的分类器，并通过对数据的权重不同使得error不同
    来影响阈值的选择，从而实现训练出将不同数据视作有不同阈值的分类器
    return bestStump: dict, 包含了哪个维度，什么阈值
           minError: double, 最小的误差
           bestClassEst: ndarray, 误差最小时的分类结果
    """
    dataMatrix = np.mat(dataArr)
    labelMat = np.mat(classLabels).T
    # 数据的行，列
    m, n = np.shape(dataMatrix)
    # 检索最佳阈值的次数
    numSteps = 10.0
    # 记录的不同的分类器
    bestStump = {}
    # 最佳的分类结果
    bestClassEst = np.mat(np.zeros((m, 1)))
    minError = 99999
    for i in range(n):
        # 找到第i列的最小值
        rangeMin = dataMatrix[:, i].min()
        # 找到第i列的最大值
        rangeMax = dataMatrix[:, i].max()
        # 对于这个特征，从最小值到最大值，分十次numSteps来查找最好的分类阈值
        stepSize = (rangeMax - rangeMin)/numSteps
        for j in range(-1, int(numSteps)+1):
            for inequal in ['lt', 'gt']:
                # 查看当前阈值的分类效果
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                # 每个样本是否分类错误
                errArr = np.mat(np.ones((m, 1)))
                errArr[predictedVals == labelMat] = 0
                # 计算加权错误，每训练一个分类器
                weightedError = D.T * errArr
                # print('split: dim {}, thresh: {}, threhs ineqal: {}, the weighted error: {}'.format(i, threshVal, inequal, weightedError))
                # 如果小于最小的error，则记录这个维度和阈值
                if weightedError < minError:
                    minError = weightedError
                    bestClassEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClassEst

def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    """
    dataArr: np.matrix, 训练数据
    classLabels: list, 标注 注意标注需要是 +1 或者-1
    numIt: int, 迭代次数
    return: weakClassArr: list, 里面包含多个set分类结果
    这个函数实现了完整的AdaBoost算法，函数名结尾的DS代表Decision Stump
    决策树桩是AdaBoost中流行的弱分类器，但并不是唯一的。
    随着迭代能被正确分类的数据的权重越来越小，所有权重之和衡为一。
    """
    weakClassArr = []
    # 多少个数据实例
    m = np.shape(dataArr)[0]
    # 初始权重
    D = np.mat(np.ones((m, 1))/m)
    # 为了记录每个数据点的类别估计累计值
    aggClassEst = np.mat(np.zeros((m, 1)))
    # 进行迭代
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        print('D: {}'.format(D.T))
        # 计算alpha来更新权重，max函数保证不会下溢
        alpha = float(0.5 * np.log((1.0 - error)/max(error, 1e-16)))
        # 记录此次的alpha
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        print('classEst: {}'.format(classEst.T))
        # 更新权重
        expon = np.multiply(-1 * alpha * np.mat(classLabels).T, classEst)
        D = D / D.sum()
        aggClassEst += alpha * classEst
        print('aggClassEst: {}'.format(aggClassEst.T))
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m, 1)))
        errorRate = aggErrors.sum() / m
        print('total error: {}'.format(errorRate))
        if errorRate == 0:
            break
    return weakClassArr

def adaClassify(datToClass, classifierArr):
	"""
	datToClass: list, 需要分类的数据
	classifierArr: list, 通过adaBoostTrainDS得到的训练结果
	这个函数利用训练好的分类器进行分类
	"""
	dataMatrix = np.mat(datToClass)
	m = np.shape(dataMatrix)[0]
	aggClassEst = np.mat(np.zeros((m, 1)))
	for i in range(len(classifierArr)):
		classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'],
			                        classifierArr[i]['thresh'],
			                        classifierArr[i]['ineq'])
		aggClassEst += classifierArr[i]['alpha']*classEst
		print(aggClassEst)
	return np.sign(aggClassEst)

def main():
    datMat, classLabels = loadSimpData()
    D = np.mat(np.ones((5, 1))/5)
    ClassifierArray = adaBoostTrainDS(datMat, classLabels, 9)
    print(ClassifierArray)
    x = adaClassify([0, 0], ClassifierArray)
    print(x)

if __name__ == '__main__':
    main()