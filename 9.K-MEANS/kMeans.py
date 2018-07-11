# coding:utf-8
import numpy as np

def loadDataSet(filename):
    """
    filename: str, 文件路径
    """
    dataMat = []
    fr = open(filename)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float, curLine)
        dataMat.append(fltLine)
    fr.close()
    return dataMat

def distEclud(vecA, vecB):
    """
    vecA: ndarray
    vecB: dnarray
    计算两个向量间的欧几里德距离
    """
    return np.sqrt(np.sum(np.power(vecA - vecB, 2)))

def randCent(dataSet, k):
    """
    dataSet: ndarray, 源数据
    k: int, k个质心
    从数据集中随机构建k个初始质心
    """
    # 每条数据有多少个属性
    n = np.shape(dataSet)[1]
    centroids = np.mat(np.zeros((k, n)))
    # 对于每个属性进行随机产生值
    for j in range(n):
        minJ = np.min(dataSet[:, j])
        rangeJ = float(np.max(dataSet[:, j]) - minJ)
        # np.random.rand(k, 1)生成k行1列的随机矩阵
        centroids[:, j] = minJ + rangeJ * np.random.rand(k, 1)
    return centroids

def kMeans(dataSet, k, distMeas=distEclud, creatCent=randCent):
    m = np.shape(dataSet)[0]
    # 聚类的结果，每条数据最终被分到了哪一类，和质心的距离是多少
    clusterAssment = np.mat(np.zeros((m, 2)))
    # 产生初始质心
    centroids = creatCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        # 对于每一条数据，计算和质心的距离
        for i in range(m):
            # 这个9999代表无穷
            minDist = 9999
            # 和minIndex质心之间的距离最短，将其归为这一类
            minIndex = 1
            # 计算和每个质心之间的距离
            for j in range(k):
                # dist between j and i
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            # clusterAssment维度是m,2 所以可以一次同时放入 这条数据是哪一类，和质心的距离是多少
            clusterAssment[i, :] = minIndex, minDist**2
        print(centroids)
        for cent in range(k):
            pstInClust = dataSet[np.nonzero(clusterAssment[:, 0].A == cent)[0]]
            centroids[cent, :] = np.mean(pstInClust, axis=0)
    return centroids, clusterAssment

def biKmeans(dataSet, k, distMeas=distEclud):
    '''
    dataSet: ndarray, 源数据
    k: int, 需要分的簇数
    distMeas: function, 衡量距离的函数
    这个函数使用二分K-MEANS聚类，先把所有的数据当成同一个簇
    然后进行二分，使得每次二分后的SSE都比原来小
    直到得到需要的簇数
    '''
    m = np.shape(dataSet)[0]
    clusterAssment = np.mat(np.zeros((m, 2))
    # 首先将所有数据点划分到同一个质心中
    centroid0 = np.mean(dataSet, axis=0).tolist()[0]
    centList = [centroid0]
    # 计算每个数据到质心的距离
    for j in range(m):
        clusterAssment[j, 1] = distMeas(np.mat(centroid0), dataSet[j, :])**2
    # 当还没有到要求的k个簇的时候就继续进行二分
    while len(centList) < k:
        # 表示无穷大
        lowestSSE = float('inf')
        # 对于每个质心再进行划分，看是否能最大程度上降低SSE
        # 每循环一次将所有簇尝试着进行二分，选择使得SSE降低最多的那个分法
        for i in range(len(centList)):
            # 目前质心包含的所有数据点，即该质心所在簇
            ptsInCurrCluster = dataSet[np.nonzero(clusterAssment[:, 0].A == i)[0], :]
            # 将簇二分
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            # 二分后的SSE
            sseSplit = np.sum(splitClustAss[:, 1])
            # 其他没有被二分的簇的SSE
            sseNoSplit = np.sum(clusterAssment[np.nonzero(clusterAssment[:,0].A!=i)[0],1]
            print('sseSplit:{}, and notSplit: {}'.format(sseSplit, sseNotSplit))
            # 当二分后的SSE变量了则更新lowestSSE，并记录此次划分的结果
            if sseSplit+sseNotSplit < lowestSSE:
                bestCentToSpit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        # 将簇分配的第多少个质心变为实际的值，因为二分的时候直接是按照0/1算的
        bestClustAss[np.nonzero(bestClustAss[:,0].A==1)[0],0]=len(centList)
        bestClustAss[np.nonzero(bestClustAss[:,0].A==0)[0],0]=bestCentToSplit
        print('The bestCentToSplit is: {}'.format(bestCentToSplit))
        print('The len of bestClustAss is: {}'.format(len(bestClustAss)))
        # 将划分结果添加到centList和clusterAssment中
        centList[bestCentToSplit]=bestNewCents[0, :]
        centList.append(bestNewCents[1, :])
        clusterAssment[np.nonzeor(clusterAssment[:,0].A==bestCentToSplit)[0],:]=bestClustAss
    return np.mat(centList), clusterAssment