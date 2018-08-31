#!user/bin/env python3
# -*- coding: gbk -*-
'''
Created on 2018年7月29日

@author: TianHeju
'''
from keras.layers.core import Dense

class autocoder(object):
    def __init__(self,encodeddimension,encoding_dim,encoding_layersnum,datadimension,inputdata):
        #输入的参数以此为：第一层编码器神经元的个数、编码器输出的最终层数、编码器的层数、数据的维度、输入的数据
        self.encodeddimension = encodeddimension
        self.encoding_dim = encoding_dim
        self.encoding_layersnum = encoding_layersnum
        self.datadimension = datadimension
        self.inputdata = inputdata
    def buildautocoder(self):
        # 压缩特征维度至self.inputdata维
        encoded = self.inputdata

        #第一层编码器神经元的个数，第一层encodeddimension，后面的比前一层的少16个
        numencodeddimension = self.encodeddimension

        # 编码层
        for i in range(0, self.encoding_layersnum):
            encoded = Dense((numencodeddimension-i*self.datadimension*8), activation='relu', name='encoded'+str(i))(encoded)
        encoder_output = Dense(self.encoding_dim, activation='relu', name='encoded'+str(self.encoding_layersnum))(encoded)
        decoded = encoder_output
        # 解码层
        for j in range(0, self.encoding_layersnum):
            decoded = Dense((numencodeddimension-(self.encoding_layersnum-j-1)*self.datadimension*8), activation='relu')(decoded)
        decoded = Dense(self.datadimension, activation='tanh')(decoded)

        return encoder_output, decoded