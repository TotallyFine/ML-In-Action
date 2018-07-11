# coding:utf-8
from sklearn import svm

# scikit-learn中的svm同时支持稠密（numpy.ndarray）和
# 稀疏矩阵（scipy.sparse）作为输入

def twoClf():
    """
    一个利用SVM进行二分类的实例
    """
    # 训练数据X需要是二维的(数据数, 特征数)X[i]即是第i条数据
    X = [[0, 0], [1, 1]]
    # 训练标签y是一维的y[i]即是第i条数据的标签
    y = [0, 1]
    # 在sklearn中svm.SVC svm.NuSVC svm.LinearSVC都可以被用来实现分类
    # SVC和NuSVC差不多，LinearSVC就直接使用线性核了不能再指定其他的核
    clf = svm.SVC()
    clf.fit(X, y)
    print(clf)
    # 使用训练好的学习器进行分类
    print(clf.predict([[2., 2.]]))
    # 可以通过support_vectors_ support_ n_support来查看支持向量
    print(clf.support_vectors_)
    print(clf.support_)
    print(clf.n_support_)

def mulClf():
    """
    多分类的实例
    """
    # SVC 和 NuSVC都可以通过一对一分类来实现多分类
    # 此时需要训练n_class*(n_class-1)/2个一对一分类器
    X = [[0], [1], [2], [3]]
    Y = [0, 1, 2, 3]
    # 通过一对一的方式实现多分类，decision_function_shape指定了使用one vs one 或者 one vs rest
    clf = svm.SVC(decision_function_shape='ovo')
    clf.fit(X, Y)
    dec = clf.decision_function([[1]])
    print(dec.shape[1]) # 4 classes 4*3/2=6
    # 下面通过指定decision_function_shape为 ovr可以实现 one vs rest
    clf.decision_function_shape = 'ovr'
    dec = clf.decision_function([[1]])
    print(dec.shape[1]) # 4 classes 4
    # LinearSVC本身就是one vs rest所以会训练n_class 个模型 如果只有两个类别就只训练一个模型
    lin_clf = svm.LinearSVC()
    lin_clf.fit(X, Y)
    print(lin_clf)
    dec = lin_clf.decision_function([[1]])
    print(dec.shape[1]) # 4
    # LinearSVC也实现了其他的ovr策略
    # Crammer and Singer方法可以通过LinearSVC(multi_class='crammer_singer')使用
    # 在实现中最常用的还是ovr，因为它的训练时间短
    
    # SVM本身不能输出概率，但是当实例化分类器的时候指定SVC(probability=True)
    # 会在结果完成之后使用Logistic回归5折交叉验证来得到概率，但是这样开销很大
    
    # 数据不平衡问题
    # SVC（不包括NuSVC）在fit方法中实现了class_weight参数
    # 这个参数需要是一个字典{class_label: value}value需要是一个大于零的浮点数
    # SVC NuSVC SVR NuSVR OneClassSVM都在fit方法中实现了sample_weight
    # 使用与class_weight类似

def svmReg():
    """
    使用SVM进行回归
    """
    # 和SVM用于分类一样，当SVM用于回归的时候只关心一小部分数据，
    # 回归的时候也是通过这一小部分数据来计算loss
    # 在sklearn中有SVR NuSVR LinearSVR
    # LinearSVR更快，只考虑了线性核。
    # 当数据是非线性的时候回归效果RBF>poly>linear
    X = [[0, 0], [2,2]]
    y = [0.5, 2.5]
    reg = svm.SVR()
    reg.fit(X, y)
    print(reg)
    print(reg.predict([[1, 1]]))

if __name__ == '__main__':
    mulClf()