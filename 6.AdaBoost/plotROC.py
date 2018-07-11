# coding:utf-8
import numpy as np

def plotROC(predStrengths, classLabels):
	"""
	predStrengths:ndarray 预测强度，即不同阈值下预测值在AdaBoost中是输入sign函数的参数
	classLabels: list, 数据的标注，值为1或其他，1认为是正例其他认为是反例
	此函数用于绘制ROC曲线，分类器会将样例分为正反两类
	给定m个正例子，n个反例子，根据学习器预测结果进行排序，
	先把分类阈值设为最大，使得所有例子均预测为反例，
	此时TPR和FPR均为0，在（0，0）处标记一个点，
	再将分类阈值依次设为每个样例的预测值，即依次将每个例子划分为正例。
	设前一个坐标为(x,y)，若当前为真正例，对应标记点为(x,y+1/m) 往上走
	若当前为假正例，则标记点为（x+1/n,y）往右走，然后依次连接各点。
    下面举个绘图例子： 有10个样例子，5个正例子，5个反例子。
    有两个学习器A,B，分别对10个例子进行预测，
    按照预测的值（这里就不具体列了）从高到低排序结果如下：
    A：[反正正正反反正正反反]
    B : [反正反反反正正正正反]
    按照绘图过程，可以得到学习器对应的ROC曲线点
    A：y:[0,0,0.2,0.4,0.6,0.6,0.6,0.8,1,1,1]
       x:[0,0.2,0.2,0.2,0.2,0.4,0.6,0.6,0.6,0.8,1]
    B：y:[0,0,0.2,0.2,0.2,0.2,0.4,0.6,0.8,1,1]
       x:[0,0.2,0.2,0.4,0.6,0.8,0.8,0.8,0.8,0.8,1]
	"""
	import matplotlib.pyplot as plt
	# 绘制光标的位置
	cur = (1.0, 1.0)
	ySum = 0.0
	# 正例的数目
	numPosClas = np.sum(np.array(classLabels)==1.0)
	# Y轴是真阳率，所以有多少个正例就有多少个y轴坐标
	yStep = 1/float(numPosClas)
	# X轴是假阳率，xStep是反例的个数分之一
	xStep = 1/float(len(classLabels)-numPosClas)
	# 得到排序索引
	sortedIndicies = predStrengths.argsort()
	fig = plt.figure()
	fir.clf()
	ax = plt.subplot(111)
	for index in sortedIndicies.tolist()[0]:
		# 遇到真正例往上走
		if classLabels[index] == 1.0:
			delX = 0
			delY = yStep
		# 遇到假正例往右走
		else:
			delX = xStep
			delY = 0
		ax.plot([cur[0], cur[0]-delX], [cur[1], cur[1]-delY], c='b')
		cur = (cur[0]-delX, cur[1]-delY)
	ax.plot([0,1], [0,1], 'b--')
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC curve for xxx System')
	ax.axis([0,1,0,1])
	plt.show()
	print('The Area Under the Curve is: {}'.format(ySum*xStep))