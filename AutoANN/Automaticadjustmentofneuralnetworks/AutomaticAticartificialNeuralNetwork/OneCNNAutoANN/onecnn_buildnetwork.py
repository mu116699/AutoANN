#!user/bin/env python3
# -*- coding: gbk -*-
'''
Created on 2018��8��1��

@author: TianHeju
'''
from keras.layers import Input
from keras.layers.core import Flatten, Dense, Dropout,Activation
from keras.layers.normalization import  BatchNormalization  as bn
from keras.layers.pooling import MaxPooling1D as pool
from keras.layers.convolutional import Conv1D as cnn1
from keras import initializers
from keras import regularizers

class onecnn_network():
    def __init__(self,layernum,featuremapnum,Convolutionkernel,input):
        #����Ĳ�������Ϊ��������Ĳ�����featuremap�ĸ���Ϊ���������2��������˵Ĵ�С
        self.layernum = layernum
        self.featuremapnum = featuremapnum
        self.Convolutionkernel = Convolutionkernel
        self.input = input
    def onecnn_buildnetwork(self):
        x = self.input
        for i in range(0, self.layernum):
            # ʹ��Ĭ�ϵĲ���kernel_initializer='random_normal',��׼��stddev
            # a layer instance is callable on a tensor, and returns a tensor
            x = cnn1(self.featuremapnum*(2**i), self.Convolutionkernel, kernel_regularizer=regularizers.l2(0.01),
                      kernel_initializer=initializers.random_normal(stddev=0.01),
                      bias_initializer='zeros')(x)
            x = bn()(x)
            x = Activation('relu')(x)
            x = Dropout(0.25)(x)
            x = pool()(x)#Ĭ�ϵĲ���Ϊ2
        x = Flatten()(x)
        x = Dense(self.featuremapnum*(2**i))(x)
        x = bn()(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(1, activation='linear')(x)
        return predictions



