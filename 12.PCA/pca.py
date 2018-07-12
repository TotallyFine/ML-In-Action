# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt

def loadDataSet(fileName, delim='\t'):
    """
    fileName: str, 文件路径
    delim: str, 分割符
    """
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [[i for i in map(float, line)] for line in stringArr]
    return np.mat(datArr)

def pca(dataMat, topNfeat=9999999):
    """
    dataMat: np.matrix, 数据
    topNfeat: int, 取前多少个特征，如果不指定的话会返回前9999999个特征或者全部特征
    """
    # 计算均值
    meanVals = np.mean(dataMat, axis=0)
    # 计算协方差矩阵
    meanRemoved = dataMat - meanVals
    covMat = np.cov(meanRemoved, rowvar=0)
    # 计算特征值
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))
    # 特征值进行从小到达排序
    eigValInd = np.argsort(eigVals)
    # 得到前topNfeat个特征
    print(eigValInd.shape)
    eigValInd = eigValInd[:-(topNfeat+1):-1]
    redEigVects = eigVects[:, eigValInd]
    # 利用N个特征将原始数据转换到新的空间中
    lowDDataMat = meanRemoved * redEigVects
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat, reconMat

def testPCA():
    dataMat = loadDataSet('testSet.txt')
    print(len(dataMat))
    lowDMat, reconMat = pca(dataMat, topNfeat=1)
    print(lowDMat.shape)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:, 0].flatten().A[0], dataMat[:, 1].flatten().A[0], marker='^', s=90)
    ax.scatter(reconMat[:, 0].flatten().A[0], reconMat[:, 1].flatten().A[0], marker='o', s=50, c='red')
    plt.show()

def replaceNanWithMean():
    """
    读取数据并用均值代替NaN
    """
    datMat = loadDataSet('secom.data', ' ')
    numFeat = np.shape(datMat)[1]
    # 对于第i个特征进行替换
    for i in range(numFeat):
        # .A是转化为ndarray，因为isnan只对ndarray起作用
        # 先通过isnan得到一个bool向量，True的是nanFalse的不是nan
        # datMat[np.isnan(datMat[:, i].A)==False, i]可以得到非nan数据
        # nonzero(isnan(datMat[:, i].A))可以得到那些nan数据的下标
        ind = np.squeeze(np.isnan(datMat[:, i].A))
        meanVal = np.mean(datMat[ind==False, i])
        datMat[ind, i] = meanVal
    return datMat

def testSecom():
    dataMat = replaceNanWithMean()
    meanVals = np.mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals
    #print(np.sum(np.isnan(meanRemoved)))
    covMat = np.cov(meanRemoved, rowvar=0)
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))
    #print(eigVals)
    # 通过下图可以看到前6个主成份覆盖了数据96.8%的方差
    # 前20覆盖了99.3%的方差，如果保留前6个就可以实现大概100：1的压缩比
    plt.plot(range(20), eigVals[:20])
    plt.show()

if __name__ == '__main__':
    testSecom()