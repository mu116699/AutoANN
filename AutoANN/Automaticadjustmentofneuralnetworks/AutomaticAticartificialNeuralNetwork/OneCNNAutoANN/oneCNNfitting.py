#!user/bin/env python3
# -*- coding: gbk -*-
'''
Created on 2018年8月1日

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
        # 输入的参数依此为数据的输入维度、神经元的层数、featuremap的个数为（2-64、输入的参数为2-65），卷积核的大小
        # 每层神经元的个数的倍率Number_of_neurons_per_layers为神经元个数的1/Cardinalnumber
        #卷积神经网络的层数也是不超过两层
        self.numdimension = numdimension
        self.layernum = layernum
        self.featuremapnum = featuremapnum
        self.Convolutionkernel = Convolutionkernel

    def trainonecnnAnn(self):
        # This returns a tensor
        inputs = Input(shape=(self.numdimension, 1))


        # fullconnection的循环参数是从1开始的，所以当神经元的个数到992需要的神经元的循环数为62+1；
        for i in range(1, self.layernum):
            for j in range(1, self.featuremapnum):
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

                # 加载数据
                # 训练参数的计算

                if i == 1:
                    # 计算公式为：F*(C+1)+F*2+((X-C+1)/2)*F*F+F+2*F+F+1
                    F = j
                    C = self.Convolutionkernel
                    X = self.numdimension
                    Trainableparams = F*(C+1)+F*2+((X-C+1)/2)*F*F+F+2*F+F+1
                    #print(Trainableparams)
                else:
                    # 计算公式为：F*(C+1)+F*2+F2*(F*C+1)+F2*2+(((X-C+1)/2-C+1)/2)*F2*F2+F2+2*F2+F2+1
                    F = j
                    C = self.Convolutionkernel
                    X = self.numdimension
                    F2 = F*2
                    Trainableparams = F*(C+1)+F*2+F2*(F*C+1)+F2*2+(((X-C+1)/2-C+1)/2)*F2*F2+F2+2*F2+F2+1
                    #print(Trainableparams)

                # 规定训练的数据集的数量为训练参数的1.5倍，测试集为训练集的0.1倍
                # int是向下取整，math.ceil是向上取整，round是四舍五入
                trainset = int(Trainableparams * 1.5)
                testset = int(0.1 * trainset)

                # 神经网络的迭代周期与批次大小Epoch、Batch Size
                # numepoch = (1500 * (i-1) + j*10)+500
                # numepoch = (10 * i + j) * 100

                # 测试训练集
                numepoch = 1
                numbatchsize = j * 50
                # print(trainset)
                # print(testset)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



                # 根据神经网络的个数自动生成数据集
                xx = traintestset(self.numdimension, trainset, testset)
                yy = xx.buildset()
                # 加载数据
                data = np.load('/tmp/Fullconnectionannresult/ONECNN-' + 'dimension-' + str(self.numdimension) +
                               '-dataset-' + str(trainset) + 'X' + str(testset) + 'X' + str(1) + '.npz')

                train_x = data['train_x']
                test_x = data['test_x']
                train_y = data['train_y']
                test_y = data['test_y']
                # 构建神经网络
                x = onecnn_network(i, j, self.Convolutionkernel, inputs)
                x = x.onecnn_buildnetwork()




                # This creates a model that includes
                # the Input layer and three Dense layers
                model = Model(input=inputs, output=x)

                # 打印模型
                model.summary()
                # 设置tensorflow的模型的放置的文件夹
                tensorflow = TensorBoard(log_dir='/tmp/Fullconnectionannresult/logsCNN-'
                                                 +'dimension-'+str(self.numdimension)+'-Hiddenlayer-'
                                                 + str(i)+'-firstfeaturemapnum-'+str(j)
                                                 + '-Convolutionkernelsize-'+str(self.Convolutionkernel)
                                                 + '-numFClayer-' +str(j*(2**(i-1)))
                                         )

                # 把每次计算的结果保存到一个csv文件中。
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

                # 设置要使用的回调函数的例子
                callback_lists = [tensorflow, csvlog, checkpoint]  # 这个链表还可以加上checkpoint的相关的点

                model.fit(train_x, train_y, validation_data=(test_x, test_y),
                          nb_epoch=numepoch, verbose=1, callbacks=callback_lists, batch_size=numbatchsize)

                # model.save_weights('/home/etcp/szx/flower_data/third_park_predict.h5')





# q = onecnnAnn(16,3,4,3)
# yy = q.trainonecnnAnn()
