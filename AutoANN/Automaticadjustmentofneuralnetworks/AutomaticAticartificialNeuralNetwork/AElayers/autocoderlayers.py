#!user/bin/env python3
# -*- coding: gbk -*-
'''
Created on 2018��7��29��

@author: TianHeju
'''
from keras.layers.core import Dense

class autocoder(object):
    def __init__(self,encodeddimension,encoding_dim,encoding_layersnum,datadimension,inputdata):
        #����Ĳ����Դ�Ϊ����һ���������Ԫ�ĸ�������������������ղ������������Ĳ��������ݵ�ά�ȡ����������
        self.encodeddimension = encodeddimension
        self.encoding_dim = encoding_dim
        self.encoding_layersnum = encoding_layersnum
        self.datadimension = datadimension
        self.inputdata = inputdata
    def buildautocoder(self):
        # ѹ������ά����self.inputdataά
        encoded = self.inputdata

        #��һ���������Ԫ�ĸ�������һ��encodeddimension������ı�ǰһ�����16��
        numencodeddimension = self.encodeddimension

        # �����
        for i in range(0, self.encoding_layersnum):
            encoded = Dense((numencodeddimension-i*self.datadimension*8), activation='relu', name='encoded'+str(i))(encoded)
        encoder_output = Dense(self.encoding_dim, activation='relu', name='encoded'+str(self.encoding_layersnum))(encoded)
        decoded = encoder_output
        # �����
        for j in range(0, self.encoding_layersnum):
            decoded = Dense((numencodeddimension-(self.encoding_layersnum-j-1)*self.datadimension*8), activation='relu')(decoded)
        decoded = Dense(self.datadimension, activation='tanh')(decoded)

        return encoder_output, decoded