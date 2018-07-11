# coding:utf-8

def loadDataSet():
    """
    这个函数返回数据集，其中每个数字都是商品编号
    每一项可以看成一次购买的商品
    """
    return [[1,3,4], [2,3,5], [1,2,3,5], [2,5]]

def creatC1(dataSet):
    """
    dataSet: list, 数据集
    这个函数通过源数据集创建C1集合
    C1集合是大小为1的所有候选项集的集合
    Apriori首先构建集合C1，然后扫描数据来判断这些只有一个元素的项集
    是否满足最小支持度的要求，满足要求的再构成L1，L1中的元素再相互组合
    构成C2，然后再进行过滤......
    """
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            # 因为是C1所以每个物品集合中只有一个物品
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    # map是python内置函数，接受一个函数一个list
    # 并将函数依次作用在list的每个元素上
    # map函数只会返回一个map对象地址
    # 所以还需遍历这个对象中的元素构成list
    x=map(frozenset, C1)
    return [item for item in x]

def scanD(D, Ck, minSupport):
    """
    D: list, 源数据集
    Ck: set, 物品项构成的集合
    minSupport: double, 最小支持度，来筛选物品的集合
    retList: list, 被筛选出的频繁项集
    supportData: set, 每个筛选出的频繁项集的支持度
    这个函数遍历源数据集，找出在Ck中的每个项的支持度，并筛选
    通过Ck得到Lk
    """
    ssCnt = {}
    for tid in D:
        for can in Ck:
            # issubset函数判断can是不是tid的子集
            if can.issubset(tid):
                # python不能创建一个只有整数的集合，所以用列表包裹起来
                # 之前未记录过这个物品集合
                if can not in ssCnt:
                    ssCnt[can] = 1
                # 已经记录过这个物品集合
                else:
                    ssCnt[can] += 1
    numItems = float(len(D))
    retList = []
    supportData = {}
    # 一个key就是一个物品的集合，ssCnt[key]就是这个物品集合出现的次数
    for key in ssCnt:
        # 计算这个物品集合的支持度
        # 并用supportData集合来记录大于最小支持度的集合
        support = ssCnt[key] / numItems
        if support >= minSupport:
            retList.insert(0, key)
        supportData[key] = support
    return retList, supportData

def aprioriGen(Lk, k):
    """
    Lk: list, 筛选Ck得到的Lk
    k: int, 
    通过Lk进行组合得到Ck+1
    """
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1, lenLk):
            # 如果前k-2个元素相同就合并这两个集合
            # | 是python中的集合合并操作符
            # 为什么比较前k-2个元素，具体的技巧见书208页
            L1 = list(Lk[i])[:k-2]
            L2 = list(Lk[j])[:k-2]
            if L1 == L2:
                retList.append(Lk[i] | Lk[j])
    return retList

def apriori(dataSet, minSupport=0.5):
    C1 = creatC1(dataSet)
    D = [item for item in map(set, dataSet)]
    L1, supportData = scanD(D, C1, minSupport)
    # L会包含 L1 L2......
    L = [L1]
    k = 2
    # 这个循环会不停查找Lk+1直到下一个大的项集为空
    # 为空的情况是不存在满足最小支持度的项集了
    while len(L[k-2]) > 0:
        Ck = aprioriGen(L[k-2], k)
        Lk, supK = scanD(D, Ck, minSupport)
        # update是set的方法，将指定字典添加进去
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData

def generateRules(L, supportData, minConf=0.7):
    """
    L: list, 经过apriori计算的一系列频繁项集
    supportData: set, 每个频繁项集作为key，value是支持度
    minConf: double, 用于筛选的最小的置信度
    """
    bigRuleList = []
    # 从1开始，因为L[0]是单个物品作为频繁项集
    for i in range(1, len(L)):
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            if i > 1:
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList

def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    """
    这个函数用来计算置信度
    """
    prunedH = []
    for conseq in H:
        conf = supportData[freqSet]/supportData[freqSet-conseq]
        if conf >= minConf:
            print('{} --> {}, conf: {}'.format(freqSet-conseq, conseq, conf))
            brl.append((freqSet-conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH

def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    """
    生成候选规则集合
    """
    m = len(H[0])
    if len(freqSet) > (m+1):
        Hmp1 = aprioriGen(H, m+1)
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        if len(Hmp1) > 1:
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)

def main():
    dataSet = loadDataSet()
    L, suppData = apriori(dataSet)
    # L是频繁项集，suppData包含有每个频繁项集的支持度
    print(L)
    rules = generateRules(L, suppData, minConf=0.7)
    print(rules)

if __name__ == '__main__':
    main()