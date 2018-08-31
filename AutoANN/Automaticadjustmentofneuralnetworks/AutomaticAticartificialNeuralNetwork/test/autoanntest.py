#!user/bin/env python3
# -*- coding: gbk -*-
'''
Created on 2018年7月28日

@author: TianHeju
'''
import sys
sys.path.append('..')
from FCAutoANN.fullconnectionAnn import fullconnectionAnn
from OneCNNAutoANN.oneCNNfitting import onecnnAnn
# #定义神经网络计算的最高维数1-15维
# FCnumdimension = 16
#
# #第一层的神经元的个数最高为992，倍率为下面的Number_of_neurons_per_layer

#基数没有当作参数被传出来，该超参定义在fullconnectionAnn.py文件中
# Number_of_neurons_per_layer =63


#测试数据集
FCnumdimension = 4
Number_of_neurons_per_layer =2

for i in range(1, FCnumdimension):
    fc = fullconnectionAnn(i,Number_of_neurons_per_layer)
    result = fc.trainfullconnectionAnn()

#制定数据构造的维数例如寻找2维最佳神经网络的
# fc = fullconnectionAnn(2, Number_of_neurons_per_layer)
# result = fc.trainfullconnectionAnn()


#默认一维卷积神经网络的层数为2；
# featuremap的个数最多为256个
#卷积核的个数满足输入的维度设为x；个数为(x-3c+3)/4>=1,
# 默认使用的卷积核为3
#一维卷积神经网络计算的维度为16-72(包括72维)
# onecnn_numdimension = 73
# featuremapnum = 65
#numepoch = 1
onecnn_numdimension = 18
featuremapnum = 4

for i in range(16, onecnn_numdimension):
    q = onecnnAnn(i,3,featuremapnum,3)
    yy = q.trainonecnnAnn()