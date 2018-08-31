#!user/bin/env python3
# -*- coding: gbk -*-
'''
Created on 2018��7��27��

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
        #����Ĳ�������Ϊ��������Ĳ�������һ����Ԫ�ĸ������������ݵ�ռλ�����Ա�����ѵ����Ȩ��
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
                      bias_initializer='zeros', weights=self.list[i])(x)# ʹ��Ĭ�ϵĲ���kernel_initializer='random_normal',��׼��stddev
            #x = Dense(self.Neuronsnum - i * Cardinalnumber, weights=self.list[i])(x)
            #���Դ����
            #print('list',self.list[i])

            x = bn()(x)
            x = Activation('relu')(x)
            x = Dropout(0.5)(x)
        #predictions = Dense(1, activation='linear')(x)
        predictions = Dense(1, activation='linear')(x)
        #���������û�м������
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