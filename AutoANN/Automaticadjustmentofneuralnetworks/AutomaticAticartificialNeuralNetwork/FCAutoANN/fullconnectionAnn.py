#!user/bin/env python3
# -*- coding: gbk -*-
'''
Created on 2018年7月27日

@author: TianHeju
'''
from keras.layers import Input
from keras.models import Model
import numpy as np
import math
from keras.callbacks import CSVLogger
from keras.callbacks import TensorBoard,ModelCheckpoint,ReduceLROnPlateau
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sys
sys.path.append('..')
#from FCAutoANNMakeData.makedata import makedata
from FCAutoANNMakeData.makedataforAckley import traintestset
#from FCAutoANN.fullconnectionlayers import layer
from FCAutoANN.buildnetwork import network
from AElayers.autocodertrain import autocodertrain

class fullconnectionAnn(object):
    """初始化类的"""
    def __init__(self,numdimension,Number_of_neurons_per_layer):
        #输入的参数依此为数据的输入维度、每层神经元的个数的倍率、默认参数神经元的层数为2（在trainfullconnectionAnn中）
        #每层神经元的个数的倍率Number_of_neurons_per_layers为神经元个数的1/Cardinalnumber
        self.numdimension = numdimension
        self.Number_of_neurons_per_layer = Number_of_neurons_per_layer

    def trainfullconnectionAnn(self,layers = 3):
        # 默认神经元的层数为2,输入的层数为层数+1
        # This returns a tensor
        inputs = Input(shape=(self.numdimension,))
        import keras.utils
        #自定的另一个超参----------------------------------
        #------------------------------------------------

        #定义基数决定函数的神经网络每一层的神经元
        Cardinalnumber=self.numdimension*128

        # 自定的另一个超参----------------------------------
        #------------------------------------------------

        #fullconnection的循环参数是从1开始的，所以当神经元的个数到992需要的神经元的循环数为62+1；
        for i in range(1, layers):
            for j in range(1, self.Number_of_neurons_per_layer):
                # -----------------------------------------------------------------------
                # 训练参数的计算
                if i == 1:
                    # 计算公式为：【(D+1)*Y1】+【Y1*2】+【Y1+1】==>简化为(D+4)*Y1+1，
                    # D为数据的维度，Y1为第一个隐藏层的神经元个数
                    Trainableparams = (j * Cardinalnumber + Cardinalnumber) * (self.numdimension + 4) + 1
                    # print(Trainableparams)
                else:
                    # 计算公式为：【(D+1)*Y1】+【Y1*2】+【(Y1+1)*Y1】+【Y2*2】+【Y2+1】==>简化为(D+3)*Y1+(Y1+4)*Y2+1，
                    # D为数据的维度，Y1为第一个隐藏层的神经元个数，Y2为第二个隐藏层的神经元个数
                    Trainableparams = (j * Cardinalnumber + Cardinalnumber) * (self.numdimension + 3) + (j * Cardinalnumber) * (j * Cardinalnumber + 20) + 1
                    # print(Trainableparams)

                # 规定训练的数据集的数量为训练参数的1.5倍，测试集为训练集的0.1倍
                # int是向下取整，math.ceil是向上取整，round是四舍五入
                trainset = int(Trainableparams * 15)
                #testset = int(0.1 * trainset)
                testset = 200

                # 神经网络的迭代周期与批次大小Epoch、Batch Size
                #numepoch = (1500 * (i-1) + j*10)+500
                #numepoch = (10 * (i-1) + j) * 100

                # #测试训练集
                numepoch = 20

                numbatchsize = j * 50

                # print(trainset)
                # print(testset)
                # 根据神经网络的个数自动生成数据集
                xx = traintestset(self.numdimension, trainset, testset)
                yy = xx.buildset()
                # 加载数据
                data = np.load('/tmp/Fullconnectionannresult/FCNN' + 'dimension' + str(self.numdimension) +
                               'dataset' + str(trainset) + 'X' + str(testset) + '.npz')

                train_x = data['train_x']
                test_x = data['test_x']
                train_y = data['train_y']
                test_y = data['test_y']
                #---------------------------------------------------------
                #加入自编码器，微调神经网络的权重
                #autocodertrain第一层编码器神经元的个数、编码器输出的最终层数、编码器的层数、数据的维度
                autocoder = autocodertrain(j * Cardinalnumber + Cardinalnumber, j * Cardinalnumber -i*Cardinalnumber+2*Cardinalnumber, i-1,
                                           self.numdimension,train_x,test_x,numepoch,numbatchsize)

                # #测试部分现在的周期设置为1：
                # autocoder = autocodertrain(j * Cardinalnumber + Cardinalnumber, j * Cardinalnumber - i * Cardinalnumber + 32, i - 1,
                #            self.numdimension, train_x, test_x, 1, numbatchsize)

                list = autocoder.buileautocodertrain()


                #-----------------------------------------------------------
                # 构建神经网络
                x = network(i, j * Cardinalnumber + Cardinalnumber, inputs,list)
                x = x.buildnetwork(Cardinalnumber)
                #-----------------------------------------------------------
                # This creates a model that includes
                # the Input layer and three Dense layers
                model = Model(input=inputs, output=x)

                # 打印模型
                model.summary()
                filenum1 = j * Cardinalnumber + Cardinalnumber
                filenum2 = 0
                if i == 2:
                    filenum2 = j * Cardinalnumber
                # 设置tensorflow的模型的放置的文件夹
                if i == 1:
                    tensorflow = TensorBoard(log_dir='/tmp/Fullconnectionannresult/FCNNlogs-'
                                                     +'dimension-'+str(self.numdimension)+
                                                     '-Hiddenlayer-'
                                                 + str(filenum1))
                else:
                    tensorflow = TensorBoard(log_dir='/tmp/Fullconnectionannresult/FCNNlogs-'
                                                     +'dimension-'+str(self.numdimension)+
                                                     '-Hiddenlayer-'
                                     + str(filenum1) + 'x' + str(filenum2))
                # 把每次计算的结果保存到一个csv文件中。
                if i == 1:
                    filename = "/tmp/Fullconnectionannresult/FCNNresult-" +'dimension-'\
                               +str(self.numdimension)+'-Hiddenlayer-'+ str(filenum1) + ".csv"
                else:
                    filename = "/tmp/Fullconnectionannresult/FCNNresult-" +'dimension-'\
                               +str(self.numdimension)+'-Hiddenlayer-'+ str(filenum1) \
                               + 'x' + str(filenum2) + ".csv"

                csvlog = CSVLogger(filename, separator=',', append=False)


                if i == 1:
                    filepath = "/tmp/Fullconnectionannresult/FCNNloss/weights-improvement-"\
                               +'dimension-'+str(self.numdimension)+'-Hiddenlayer-'+ str(filenum1) \
                               + "-val_loss-{val_loss:.5f}-val_mae-{val_mean_absolute_error:.5f}-epoch-{epoch:04d}.hdf5"
                else:
                    filepath = "/tmp/Fullconnectionannresult/FCNNloss/" \
                               "weights-improvement-" +'dimension-'+str(self.numdimension)\
                               +'-Hiddenlayer-'+ str(filenum1) + 'X' + str(filenum2) + \
                               "-val_loss-{val_loss:.5f}-val_mae-{val_mean_absolute_error:.5f}-epoch-{epoch:04d}.hdf5"

                checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min',period=1)

                #回调函数还可以添加
                # ReduceLROnPlateau当评价指标不在提升时，减少学习率当学习停滞时，减少 2 倍或 10 倍的学习率常常能获得较好的效果。
                # 该回调函数检测指标的情况，如果在 patience 个 epoch 中看不到模型性能提升，则减少学习率
                #ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.01),


                #EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=5, verbose=1, mode='auto'),
                #CustomModelCheckpoint(self.encoder, self.save_model, monitor='val_loss', save_best_only=True, mode='auto')

                optimizers = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
                # Hinton 建议设定 γ 为 0.9, 学习率 η 为 0.001。

                model.compile(optimizer=optimizers,
                              loss='mean_squared_error',
                              metrics=['mae', 'acc'])

                # 设置要使用的回调函数的例子
                callback_lists = [tensorflow, csvlog, checkpoint]  # 这个链表还可以加上checkpoint的相关的点


                model.fit(train_x, train_y, validation_data=(test_x, test_y),
                          nb_epoch=numepoch, verbose=1, callbacks=callback_lists, batch_size=numbatchsize)
                #---------------------------------------------------------------------
                # 作出预测与真实值保存到txt文件内
                predict_y = model.predict(test_x)
                predict_y1 = predict_y.reshape(testset)
                #print('predict_y:',predict_y1)
                #print('test_y:',test_y)
                test_real_and_predict_result0 = np.concatenate([test_y,predict_y1]).reshape(2,testset)
                test_real_and_predict_result = test_real_and_predict_result0.transpose()
                #print(test_real_and_predict_result)
                if i ==1:
                    np.savetxt('/tmp/Fullconnectionannresult/Summaryofresults_test_real_and_predict_result'
                               + 'dimension-' + str(self.numdimension) +'-Hiddenlayer-'
                                + str(filenum1)+'.txt', test_real_and_predict_result)
                else:
                    np.savetxt('/tmp/Fullconnectionannresult/Summaryofresults_test_real_and_predict_result'+
                               'dimension-'+str(self.numdimension)+'-Hiddenlayer-'
                                + str(filenum1) + 'x' + str(filenum2)+'.txt',test_real_and_predict_result)
                # for i in range(len(test_x)):
                #     print('X=%s, Predicted=%s' % (test_x[i], predict_y[i]))

                #绘制最后的预测值
                test_x1 = test_x.transpose()
                #print('test_x1',test_x1)
                if self.numdimension == 1:
                    plt.figure(figsize=(8, 8))
                    plt.scatter(test_x1, test_y,c='b',marker='o')
                    plt.scatter(test_x1, predict_y,c='r',marker='^')
                    plt.xlabel('variable x')
                    plt.ylabel('variable y')
                else:
                    fig = plt.figure(figsize=(8, 8))
                    ax = fig.add_subplot(111, projection='3d')
                    ax.scatter(test_x1[0],test_x1[1],test_y,marker='o', color='blue')
                    ax.scatter(test_x1[0],test_x1[1],predict_y,marker='^', color='red')
                    ax.set_xlabel('variable x[0]')
                    ax.set_ylabel('variable x[1]')
                    ax.set_zlabel('variable y')
                if i == 1:
                    plt.title('FCNNpredict-'+'dimension-'+str(self.numdimension)+'-Hiddenlayer-'
                            + str(filenum1)+'-dataset-'+str(trainset)+'x'+str(testset)+
                              '-epoch-'+str(numepoch))
                else:
                    plt.title('FCNNpredict-'+'dimension-'+str(self.numdimension)+'-Hiddenlayer-'
                            + str(filenum1) + 'x' + str(filenum2)+'-dataset-'+str(trainset)+
                              'x'+str(testset)+'-epoch-'+str(numepoch))
                plt.show()
                #-----------------------------------------------------------------


                # ---------------------------------------------------------------------
                # 作出训练与真实值保存到txt文件内
                predict_yy = model.predict(train_x)
                predict_yy1 = predict_yy.reshape(trainset)
                # print('predict_y:',predict_y1)
                # print('test_y:',test_y)
                test_real_and_predict_result10 = np.concatenate([train_y, predict_yy1]).reshape(2, trainset)
                test_real_and_predict_result2 = test_real_and_predict_result10.transpose()
                # print(test_real_and_predict_result)
                if i == 1:
                    np.savetxt('/tmp/Fullconnectionannresult/Summaryofresults_test_real_and_predict_result'
                               + 'dimension-' + str(self.numdimension) + '-Hiddenlayer-'
                               + str(filenum1) + '.txt', test_real_and_predict_result2)
                else:
                    np.savetxt('/tmp/Fullconnectionannresult/Summaryofresults_test_real_and_predict_result' +
                               'dimension-' + str(self.numdimension) + '-Hiddenlayer-'
                               + str(filenum1) + 'x' + str(filenum2) + '.txt', test_real_and_predict_result2)
                # for i in range(len(test_x)):
                #     print('X=%s, Predicted=%s' % (test_x[i], predict_y[i]))

                # 绘制最后的预测值
                train_x1 = train_x.transpose()
                # print('test_x1',test_x1)
                if self.numdimension == 1:
                    plt.figure(figsize=(8, 8))
                    plt.scatter(train_x1, train_y, c='b', marker='o')
                    plt.scatter(train_x1, predict_yy, c='r', marker='^')
                    plt.xlabel('variable x')
                    plt.ylabel('variable y')
                else:
                    fig = plt.figure(figsize=(8, 8))
                    ax = fig.add_subplot(111, projection='3d')
                    ax.scatter(test_x1[0], test_x1[1], test_y, marker='o', color='blue')
                    ax.scatter(test_x1[0], test_x1[1], predict_y, marker='^', color='red')
                    ax.set_xlabel('variable x[0]')
                    ax.set_ylabel('variable x[1]')
                    ax.set_zlabel('variable y')
                if i == 1:
                    plt.title('FCNNpredict-' + 'dimension-' + str(self.numdimension) + '-Hiddenlayer-'
                              + str(filenum1) + '-dataset-' + str(trainset) + 'x' + str(testset) +
                              '-epoch-' + str(numepoch))
                else:
                    plt.title('FCNNpredict-' + 'dimension-' + str(self.numdimension) + '-Hiddenlayer-'
                              + str(filenum1) + 'x' + str(filenum2) + '-dataset-' + str(trainset) +
                              'x' + str(testset) + '-epoch-' + str(numepoch))
                plt.show()
                # -----------------------------------------------------------------
                # model.fit(train_x, train_y, validation_data=(test_x, test_y),
                # nb_epoch=1, verbose=1, callbacks=callback_lists, batch_size=numbatchsize)

#测试代码部分
fc = fullconnectionAnn(1,2)
result = fc.trainfullconnectionAnn()