#!user/bin/env python3
# -*- coding: gbk -*-
'''
Created on 2018��7��28��

@author: TianHeju
'''
from keras.layers import Input
from keras.models import Model
from keras.layers.core import  Dense
import numpy as np
from keras.callbacks import CSVLogger
from keras.callbacks import TensorBoard
import pandas as pd
import matplotlib.pyplot as plt
import  keras.utils

from AElayers.autocoderlayers import autocoder

# encodeddimension = 64
# encoding_dim = 10
# encoding_layersnum = 3
# datadimension = 12
# numepoch = 1
# numbatchsize = 1000

class autocodertrain(object):
    def __init__(self,encodeddimension, encoding_dim, encoding_layersnum,
                 datadimension,train_x,test_x,numepoch,numbatchsize):
        #��һ���������Ԫ�ĸ�������������������ղ������������Ĳ�����
        # ���ݵ�ά�ȡ�ѵ�����������Բ������������ڡ�����������
        self.encodeddimension = encodeddimension
        self.encoding_dim = encoding_dim
        self.encoding_layersnum = encoding_layersnum
        self.datadimension = datadimension
        self.train_x = train_x
        self.test_x = test_x
        self.numepoch = numepoch
        self.numbatchsize = numbatchsize
    def buileautocodertrain(self):
        # This returns a tensor
        inputs = Input(shape=(self.datadimension,))
        # ����������ʱ������3��Ҫ����һ�������ƶ����һ��ı����������Ԫ�ĸ���
        # ����Ĳ����Դ�Ϊ����һ���������Ԫ�ĸ�������������������ղ������������Ĳ��������ݵ�ά�ȡ����������
        AE = autocoder(self.encodeddimension, self.encoding_dim,
                       self.encoding_layersnum, self.datadimension, inputs)
        encoder_output, decoded = AE.buildautocoder()

        # �����Ա���ģ��
        autoencoder = Model(inputs=inputs, outputs=decoded)

        # ��������ģ��
        encoder = Model(inputs=inputs, outputs=encoder_output)

        # print model
        print('------------------------------------')
        print('models layers:', autoencoder.layers)
        print('models config:', autoencoder.get_config())
        print('models summary:', autoencoder.summary())
        print('------------------------------------')

        # ����tensorflow��ģ�͵ķ��õ��ļ���
        if self.encoding_layersnum == 0:
            tensorflow = TensorBoard(log_dir='/tmp/Fullconnectionannresult/logsAE-'
                                             +'dimension-'+str(self.datadimension)+
                                             '-Hiddenlayer-'+
                                             str(self.encoding_dim))
        else:
            tensorflow = TensorBoard(log_dir='/tmp/Fullconnectionannresult/logsAE-'
                                             +'dimension-'+str(self.datadimension)+
                                             '-Hiddenlayer-'
                    + str(self.encodeddimension) + 'X' + str(self.encoding_dim))

        # ��ÿ�μ���Ľ�����浽һ��csv�ļ��С�

        if self.encoding_layersnum == 0:
            filename = '/tmp/Fullconnectionannresult/resultAE-'+'dimension-'\
                       +str(self.datadimension)+'-Hiddenlayer-' + str(self.encoding_dim) + '.csv'
        else:
            filename = '/tmp/Fullconnectionannresult/resultAE-'+'dimension-'+\
                       str(self.datadimension)+'-Hiddenlayer-'+ str(self.encodeddimension) \
                       + 'X' + str(self.encoding_dim) + '.csv'

        csvlog = CSVLogger(filename, separator=',', append=False)

        # ����Ҫʹ�õĻص�����������
        callback_lists = [tensorflow, csvlog]  # ����������Լ���checkpoint����صĵ�

        optimizers = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
        # Hinton �����趨 �� Ϊ 0.9, ѧϰ�� �� Ϊ 0.001��

        # compile autoencoder
        autoencoder.compile(optimizer=optimizers, loss='mse')

        # training
        autoencoder.fit(self.train_x, self.train_x, epochs=self.numepoch,
                        callbacks=callback_lists,batch_size=self.numbatchsize, shuffle=True)

        # plotting suorce
        inputdata = self.test_x.transpose()

        if self.datadimension == 1:
            plt.scatter(inputdata[0], inputdata[0], c='blue', s=3)
        else:
            plt.scatter(inputdata[0], inputdata[1], c='blue', s=3)

        info1 = pd.DataFrame(self.test_x)
        # �����excel��
        if self.encoding_layersnum == 0:
            writer = pd.ExcelWriter('/tmp/Fullconnectionannresult/inputAE-'+'dimension-'
                                    +str(self.datadimension)+'-Hiddenlayer-'+
                                             str(self.encoding_dim)+ '.xlsx')
        else:
            writer = pd.ExcelWriter('/tmp/Fullconnectionannresult/inputAE-'+'dimension-'
                                    +str(self.datadimension)+'-Hiddenlayer-'
                    + str(self.encodeddimension) + 'X' + str(self.encoding_dim) + '.xlsx')
        info1.to_excel(writer, 'page_1', float_format='%.5f')
        writer.save()

        # output encoder
        encoder_imgs1 = encoder.predict(self.test_x)
        encoder_imgs = encoder_imgs1.transpose()
        info2 = pd.DataFrame(encoder_imgs1)
        # �����excel��
        if self.encoding_layersnum == 0:
            writer = pd.ExcelWriter('/tmp/Fullconnectionannresult/encoderoutputAE-'+
                                    'dimension-'+str(self.datadimension)+'-Hiddenlayer-'+
                                             str(self.encoding_dim)+ '.xlsx')
        else:
            writer = pd.ExcelWriter('/tmp/Fullconnectionannresult/encoderoutputAE-'+
                                    'dimension-'+str(self.datadimension)+'-Hiddenlayer-'
                    + str(self.encodeddimension) + 'X' + str(self.encoding_dim) + '.xlsx')
        info2.to_excel(writer, 'page_1', float_format='%.5f')
        writer.save()

        # output autoencoder and ploting
        autoencoder_imgs1 = autoencoder.predict(self.test_x)
        autoencoder_imgs = autoencoder_imgs1.transpose()

        if self.datadimension == 1:
            plt.scatter(autoencoder_imgs[0], autoencoder_imgs[0], c='red', s=3)
        else:
            plt.scatter(autoencoder_imgs[0], autoencoder_imgs[1], c='red', s=3)

        info3 = pd.DataFrame(autoencoder_imgs1)
        # �����excel��
        if self.encoding_layersnum == 0:
            writer = pd.ExcelWriter('/tmp/Fullconnectionannresult/autoencoderoutputAE-'
                                    +'dimension-'+str(self.datadimension)+'-Hiddenlayer-'+
                                             str(self.encoding_dim)+ '.xlsx')
        else:
            writer = pd.ExcelWriter('/tmp/Fullconnectionannresult/autoencoderoutputAE-'
                                    +'dimension-'+str(self.datadimension)+'-Hiddenlayer-'
                    + str(self.encodeddimension) + 'X' + str(self.encoding_dim) + '.xlsx')
        info3.to_excel(writer, 'page_1', float_format='%.5f')
        writer.save()

        plt.title('Autoencoder run' + str(self.numepoch) + 'time')
        plt.xlabel('variables x[0]')
        plt.ylabel('variables x[1]')

        #����������ϵĶԱȽ������������ϼ���
        if self.encoding_layersnum == 0:
            plt.savefig('/tmp/Fullconnectionannresult/Autoencoder-'+'dimension-'
                        +str(self.datadimension)+'-Hiddenlayer-'+
                                             str(self.encoding_dim)+ '.png')
        else:
            plt.savefig('/tmp/Fullconnectionannresult/Autoencoder-'+'dimension-'
                        +str(self.datadimension)+'-Hiddenlayer-'
                    + str(self.encodeddimension) + 'X' + str(self.encoding_dim) + '.png')
        # plt.show()
        plt.close()
        #����Ա�������Ȩ��
        list = []
        for i in range(0, self.encoding_layersnum + 1):
            # get layers by name
            names = locals()
            names['encoded%s' % i] = autoencoder.get_layer(name='encoded' + str(i))
            names['layerweight%s' % i] = names['encoded%s' % i].get_weights()
            list.append(names['layerweight%s' % i])

        #���Դ����
        #print('rellist',list)

        return list
        # print(layerweight3)
        # print(layerweight0)







