#!user/bin/env python3
# -*- coding: gbk -*-
'''
Created on 2018年7月27日

@author: TianHeju
'''
#from FCAutoANN.layers import layer
from keras.layers.core import Dense, Activation
from keras.layers import Input
from keras.layers.core import Dense,Activation, Dropout
from keras.layers.normalization import  BatchNormalization  as bn
from keras import initializers
from keras import regularizers

class network():
    def __init__(self,layernum,Neuronsnum,input,list):
        #输入的参数依此为：神经网络的层数、第一层神经元的个数，输入数据的占位符、自编码器训练的权重
        self.layernum = layernum
        self.Neuronsnum = Neuronsnum
        self.input = input
        self.list = list

    def buildnetwork(self,Cardinalnumber):
        x = self.input
        for i in range(0, self.layernum):
            x = Dense(self.Neuronsnum-i*Cardinalnumber,
                      kernel_regularizer=regularizers.l2(0.01),
                      kernel_initializer=initializers.random_normal(stddev=0.01),
                      bias_initializer='zeros', weights=self.list[i])(x)# 使用默认的参数kernel_initializer='random_normal',标准差stddev
            #x = Dense(self.Neuronsnum - i * Cardinalnumber, weights=self.list[i])(x)
            #测试代码段
            #print('list',self.list[i])

            x = bn()(x)
            x = Activation('relu')(x)
            x = Dropout(0.5)(x)
        #predictions = Dense(1, activation='linear')(x)
        predictions = Dense(1, activation='linear')(x)
        #最后的输出是没有激活函数的
        #predictions = Dense(1)(x)
        return predictions


# from FCAutoANN.layers import layer
# from keras.layers.core import Dense, Activation
#
# class network():
#     def __init__(self,layernum,Neuronsnum,input):
#         self.layernum = layernum
#         self.Neuronsnum = Neuronsnum
#         self.input = input
#
#     def buildnetwork(self,list):
#         x = self.input
#         for i in range(0, self.layernum):
#             layers = layer(self.Neuronsnum-i*16,x,list)
#             x = layers.buildlayer()
#         predictions = Dense(1, activation='linear')(x)
#         return predictions