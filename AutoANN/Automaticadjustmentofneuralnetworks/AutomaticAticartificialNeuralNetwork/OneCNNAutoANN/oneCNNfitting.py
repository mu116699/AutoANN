#!user/bin/env python3
# -*- coding: gbk -*-
'''
Created on 2018��8��1��

@author: TianHeju
'''
from keras.layers import Input
from keras.models import Model
from keras.layers.core import Flatten, Dense, Dropout,Activation
from keras.layers.normalization import  BatchNormalization  as bn
from keras.layers.pooling import MaxPooling1D as pool
from keras.layers.convolutional import Conv1D as cnn1
from keras.callbacks import CSVLogger, TensorBoard, ModelCheckpoint
import numpy as np

import sys
sys.path.append('..')
#from FCAutoANNMakeData.makedata import makedata
from OneCNNAutoANNMakeData.makedataforAckley import traintestset
from OneCNNAutoANN.onecnn_buildnetwork import onecnn_network


class onecnnAnn(object):
    def __init__(self,numdimension,layernum,featuremapnum,Convolutionkernel):
        # ����Ĳ�������Ϊ���ݵ�����ά�ȡ���Ԫ�Ĳ�����featuremap�ĸ���Ϊ��2-64������Ĳ���Ϊ2-65��������˵Ĵ�С
        # ÿ����Ԫ�ĸ����ı���Number_of_neurons_per_layersΪ��Ԫ������1/Cardinalnumber
        #���������Ĳ���Ҳ�ǲ���������
        self.numdimension = numdimension
        self.layernum = layernum
        self.featuremapnum = featuremapnum
        self.Convolutionkernel = Convolutionkernel

    def trainonecnnAnn(self):
        # This returns a tensor
        inputs = Input(shape=(self.numdimension, 1))


        # fullconnection��ѭ�������Ǵ�1��ʼ�ģ����Ե���Ԫ�ĸ�����992��Ҫ����Ԫ��ѭ����Ϊ62+1��
        for i in range(1, self.layernum):
            for j in range(1, self.featuremapnum):
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

                # ��������
                # ѵ�������ļ���

                if i == 1:
                    # ���㹫ʽΪ��F*(C+1)+F*2+((X-C+1)/2)*F*F+F+2*F+F+1
                    F = j
                    C = self.Convolutionkernel
                    X = self.numdimension
                    Trainableparams = F*(C+1)+F*2+((X-C+1)/2)*F*F+F+2*F+F+1
                    #print(Trainableparams)
                else:
                    # ���㹫ʽΪ��F*(C+1)+F*2+F2*(F*C+1)+F2*2+(((X-C+1)/2-C+1)/2)*F2*F2+F2+2*F2+F2+1
                    F = j
                    C = self.Convolutionkernel
                    X = self.numdimension
                    F2 = F*2
                    Trainableparams = F*(C+1)+F*2+F2*(F*C+1)+F2*2+(((X-C+1)/2-C+1)/2)*F2*F2+F2+2*F2+F2+1
                    #print(Trainableparams)

                # �涨ѵ�������ݼ�������Ϊѵ��������1.5�������Լ�Ϊѵ������0.1��
                # int������ȡ����math.ceil������ȡ����round����������
                trainset = int(Trainableparams * 1.5)
                testset = int(0.1 * trainset)

                # ������ĵ������������δ�СEpoch��Batch Size
                # numepoch = (1500 * (i-1) + j*10)+500
                # numepoch = (10 * i + j) * 100

                # ����ѵ����
                numepoch = 1
                numbatchsize = j * 50
                # print(trainset)
                # print(testset)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



                # ����������ĸ����Զ��������ݼ�
                xx = traintestset(self.numdimension, trainset, testset)
                yy = xx.buildset()
                # ��������
                data = np.load('/tmp/Fullconnectionannresult/ONECNN-' + 'dimension-' + str(self.numdimension) +
                               '-dataset-' + str(trainset) + 'X' + str(testset) + 'X' + str(1) + '.npz')

                train_x = data['train_x']
                test_x = data['test_x']
                train_y = data['train_y']
                test_y = data['test_y']
                # ����������
                x = onecnn_network(i, j, self.Convolutionkernel, inputs)
                x = x.onecnn_buildnetwork()




                # This creates a model that includes
                # the Input layer and three Dense layers
                model = Model(input=inputs, output=x)

                # ��ӡģ��
                model.summary()
                # ����tensorflow��ģ�͵ķ��õ��ļ���
                tensorflow = TensorBoard(log_dir='/tmp/Fullconnectionannresult/logsCNN-'
                                                 +'dimension-'+str(self.numdimension)+'-Hiddenlayer-'
                                                 + str(i)+'-firstfeaturemapnum-'+str(j)
                                                 + '-Convolutionkernelsize-'+str(self.Convolutionkernel)
                                                 + '-numFClayer-' +str(j*(2**(i-1)))
                                         )

                # ��ÿ�μ���Ľ�����浽һ��csv�ļ��С�
                filename = '/tmp/Fullconnectionannresult/resultCNN-'+'dimension-'+str(self.numdimension)\
                                                +'-Hiddenlayer-'+str(i)+'-featuremapnum-'+str(j)+ \
                           '-Convolutionkernelsize-'+str(self.Convolutionkernel)+ '-FClayer-' \
                           +str(j*(2**(i-1)))+'.csv'
                csvlog = CSVLogger(filename, separator=',', append=False)

                filepath = "/tmp/Fullconnectionannresult/FCNNloss/" \
                           "weights-improvement-" + 'dimension-' + str(self.numdimension) \
                           + '-Hiddenlayer-' + str(i) + '-featuremapnum-' + str(j) + \
                           '-Convolutionkernelsize-' + str(self.Convolutionkernel) + '-FClayer-' \
                           + str(j * (2 ** (i - 1)))+\
                           "-{epoch:04d}-{val_loss:.5f}.hdf5"

                checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min',
                                             period=1)

                model.compile(optimizer='rmsprop',
                              loss='mean_squared_error',
                              metrics=['mae', 'acc'])

                # ����Ҫʹ�õĻص�����������
                callback_lists = [tensorflow, csvlog, checkpoint]  # ����������Լ���checkpoint����صĵ�

                model.fit(train_x, train_y, validation_data=(test_x, test_y),
                          nb_epoch=numepoch, verbose=1, callbacks=callback_lists, batch_size=numbatchsize)

                # model.save_weights('/home/etcp/szx/flower_data/third_park_predict.h5')





# q = onecnnAnn(16,3,4,3)
# yy = q.trainonecnnAnn()
