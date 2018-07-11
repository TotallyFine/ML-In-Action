# coding:utf-8
# 对马得了疝气是否能存活进行判断

import numpy as np
from log_regress import stoc_grad_ascent1, classify_vector

def cloic_test():
    # 打开文件
    fr_train = open('horse_colic_training.txt')
    fr_test = open('horse_colic_test.txt')
    training_set = []
    training_labels = []
    # 从文件中读取数据，按行解析
    for line in fr_train.readlines():
        curr_line = line.strip().split('\t')
        line_arr = []
        # 1-21列是特征，第22行是标签
        for i in range(21):
            line_arr.append(float(curr_line[i]))
        training_set.append(line_arr)
        training_labels.append(float(curr_line[21]))
    # 进行训练
    train_weights = stoc_grad_ascent1(np.array(training_set), training_labels, num_iter=500)
    # 预测结果不符合实际的数目
    error_count = 0
    # 测试数据数目
    num_test_vec = 0.0
    for line in fr_test.readlines():
        num_test_vec += 1
        curr_line = line.strip().split('\t')
        line_arr = []
        for i in range(21):
            line_arr.append(float(curr_line[i]))
        if int(classify_vector(np.array(line_arr), train_weights)) != int(float(curr_line[21])):
            error_count +=1 
    error_rate = float(error_count)/num_test_vec
    print('the error rate of test is: {}'.format(error_rate))
    return error_rate

def multi_test():
    num_test = 10
    error_sum = 0.0
    for k in range(num_test):
        error_sum += cloic_test()
    print('after {} iterations the average error rate is: {}'.format(num_test, error_sum/float(num_test)))

if __name__ == '__main__':
    #the error rate of test is: 0.28762541806020064
    #the error rate of test is: 0.3411371237458194
    #the error rate of test is: 0.33444816053511706
    #the error rate of test is: 0.3612040133779264
    #the error rate of test is: 0.27424749163879597
    #the error rate of test is: 0.3511705685618729
    #the error rate of test is: 0.4013377926421405
    #the error rate of test is: 0.32441471571906355
    #the error rate of test is: 0.3277591973244147
    #the error rate of test is: 0.3076923076923077
    #after 10 iterations the average error rate is: 0.3311036789297659
    multi_test()