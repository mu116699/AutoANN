#!user/bin/env python3
# -*- coding: gbk -*-
'''
Created on 2018��7��28��

@author: TianHeju
'''
import sys
sys.path.append('..')
from FCAutoANN.fullconnectionAnn import fullconnectionAnn
from OneCNNAutoANN.oneCNNfitting import onecnnAnn
# #�����������������ά��1-15ά
# FCnumdimension = 16
#
# #��һ�����Ԫ�ĸ������Ϊ992������Ϊ�����Number_of_neurons_per_layer

#����û�е������������������ó��ζ�����fullconnectionAnn.py�ļ���
# Number_of_neurons_per_layer =63


#�������ݼ�
FCnumdimension = 4
Number_of_neurons_per_layer =2

for i in range(1, FCnumdimension):
    fc = fullconnectionAnn(i,Number_of_neurons_per_layer)
    result = fc.trainfullconnectionAnn()

#�ƶ����ݹ����ά������Ѱ��2ά����������
# fc = fullconnectionAnn(2, Number_of_neurons_per_layer)
# result = fc.trainfullconnectionAnn()


#Ĭ��һά���������Ĳ���Ϊ2��
# featuremap�ĸ������Ϊ256��
#����˵ĸ������������ά����Ϊx������Ϊ(x-3c+3)/4>=1,
# Ĭ��ʹ�õľ����Ϊ3
#һά�������������ά��Ϊ16-72(����72ά)
# onecnn_numdimension = 73
# featuremapnum = 65
#numepoch = 1
onecnn_numdimension = 18
featuremapnum = 4

for i in range(16, onecnn_numdimension):
    q = onecnnAnn(i,3,featuremapnum,3)
    yy = q.trainonecnnAnn()