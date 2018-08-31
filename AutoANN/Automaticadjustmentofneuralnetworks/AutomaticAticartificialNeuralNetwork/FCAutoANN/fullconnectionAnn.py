#!user/bin/env python3
# -*- coding: gbk -*-
'''
Created on 2018��7��27��

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
    """��ʼ�����"""
    def __init__(self,numdimension,Number_of_neurons_per_layer):
        #����Ĳ�������Ϊ���ݵ�����ά�ȡ�ÿ����Ԫ�ĸ����ı��ʡ�Ĭ�ϲ�����Ԫ�Ĳ���Ϊ2����trainfullconnectionAnn�У�
        #ÿ����Ԫ�ĸ����ı���Number_of_neurons_per_layersΪ��Ԫ������1/Cardinalnumber
        self.numdimension = numdimension
        self.Number_of_neurons_per_layer = Number_of_neurons_per_layer

    def trainfullconnectionAnn(self,layers = 3):
        # Ĭ����Ԫ�Ĳ���Ϊ2,����Ĳ���Ϊ����+1
        # This returns a tensor
        inputs = Input(shape=(self.numdimension,))
        import keras.utils
        #�Զ�����һ������----------------------------------
        #------------------------------------------------

        #�����������������������ÿһ�����Ԫ
        Cardinalnumber=self.numdimension*128

        # �Զ�����һ������----------------------------------
        #------------------------------------------------

        #fullconnection��ѭ�������Ǵ�1��ʼ�ģ����Ե���Ԫ�ĸ�����992��Ҫ����Ԫ��ѭ����Ϊ62+1��
        for i in range(1, layers):
            for j in range(1, self.Number_of_neurons_per_layer):
                # -----------------------------------------------------------------------
                # ѵ�������ļ���
                if i == 1:
                    # ���㹫ʽΪ����(D+1)*Y1��+��Y1*2��+��Y1+1��==>��Ϊ(D+4)*Y1+1��
                    # DΪ���ݵ�ά�ȣ�Y1Ϊ��һ�����ز����Ԫ����
                    Trainableparams = (j * Cardinalnumber + Cardinalnumber) * (self.numdimension + 4) + 1
                    # print(Trainableparams)
                else:
                    # ���㹫ʽΪ����(D+1)*Y1��+��Y1*2��+��(Y1+1)*Y1��+��Y2*2��+��Y2+1��==>��Ϊ(D+3)*Y1+(Y1+4)*Y2+1��
                    # DΪ���ݵ�ά�ȣ�Y1Ϊ��һ�����ز����Ԫ������Y2Ϊ�ڶ������ز����Ԫ����
                    Trainableparams = (j * Cardinalnumber + Cardinalnumber) * (self.numdimension + 3) + (j * Cardinalnumber) * (j * Cardinalnumber + 20) + 1
                    # print(Trainableparams)

                # �涨ѵ�������ݼ�������Ϊѵ��������1.5�������Լ�Ϊѵ������0.1��
                # int������ȡ����math.ceil������ȡ����round����������
                trainset = int(Trainableparams * 15)
                #testset = int(0.1 * trainset)
                testset = 200

                # ������ĵ������������δ�СEpoch��Batch Size
                #numepoch = (1500 * (i-1) + j*10)+500
                #numepoch = (10 * (i-1) + j) * 100

                # #����ѵ����
                numepoch = 20

                numbatchsize = j * 50

                # print(trainset)
                # print(testset)
                # ����������ĸ����Զ��������ݼ�
                xx = traintestset(self.numdimension, trainset, testset)
                yy = xx.buildset()
                # ��������
                data = np.load('/tmp/Fullconnectionannresult/FCNN' + 'dimension' + str(self.numdimension) +
                               'dataset' + str(trainset) + 'X' + str(testset) + '.npz')

                train_x = data['train_x']
                test_x = data['test_x']
                train_y = data['train_y']
                test_y = data['test_y']
                #---------------------------------------------------------
                #�����Ա�������΢���������Ȩ��
                #autocodertrain��һ���������Ԫ�ĸ�������������������ղ������������Ĳ��������ݵ�ά��
                autocoder = autocodertrain(j * Cardinalnumber + Cardinalnumber, j * Cardinalnumber -i*Cardinalnumber+2*Cardinalnumber, i-1,
                                           self.numdimension,train_x,test_x,numepoch,numbatchsize)

                # #���Բ������ڵ���������Ϊ1��
                # autocoder = autocodertrain(j * Cardinalnumber + Cardinalnumber, j * Cardinalnumber - i * Cardinalnumber + 32, i - 1,
                #            self.numdimension, train_x, test_x, 1, numbatchsize)

                list = autocoder.buileautocodertrain()


                #-----------------------------------------------------------
                # ����������
                x = network(i, j * Cardinalnumber + Cardinalnumber, inputs,list)
                x = x.buildnetwork(Cardinalnumber)
                #-----------------------------------------------------------
                # This creates a model that includes
                # the Input layer and three Dense layers
                model = Model(input=inputs, output=x)

                # ��ӡģ��
                model.summary()
                filenum1 = j * Cardinalnumber + Cardinalnumber
                filenum2 = 0
                if i == 2:
                    filenum2 = j * Cardinalnumber
                # ����tensorflow��ģ�͵ķ��õ��ļ���
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
                # ��ÿ�μ���Ľ�����浽һ��csv�ļ��С�
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

                #�ص��������������
                # ReduceLROnPlateau������ָ�겻������ʱ������ѧϰ�ʵ�ѧϰͣ��ʱ������ 2 ���� 10 ����ѧϰ�ʳ����ܻ�ýϺõ�Ч����
                # �ûص��������ָ������������� patience �� epoch �п�����ģ�����������������ѧϰ��
                #ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.01),


                #EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=5, verbose=1, mode='auto'),
                #CustomModelCheckpoint(self.encoder, self.save_model, monitor='val_loss', save_best_only=True, mode='auto')

                optimizers = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
                # Hinton �����趨 �� Ϊ 0.9, ѧϰ�� �� Ϊ 0.001��

                model.compile(optimizer=optimizers,
                              loss='mean_squared_error',
                              metrics=['mae', 'acc'])

                # ����Ҫʹ�õĻص�����������
                callback_lists = [tensorflow, csvlog, checkpoint]  # ����������Լ���checkpoint����صĵ�


                model.fit(train_x, train_y, validation_data=(test_x, test_y),
                          nb_epoch=numepoch, verbose=1, callbacks=callback_lists, batch_size=numbatchsize)
                #---------------------------------------------------------------------
                # ����Ԥ������ʵֵ���浽txt�ļ���
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

                #��������Ԥ��ֵ
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
                # ����ѵ������ʵֵ���浽txt�ļ���
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

                # ��������Ԥ��ֵ
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

#���Դ��벿��
fc = fullconnectionAnn(1,2)
result = fc.trainfullconnectionAnn()