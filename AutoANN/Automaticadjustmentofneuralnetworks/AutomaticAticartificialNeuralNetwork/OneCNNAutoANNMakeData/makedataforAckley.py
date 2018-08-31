#!user/bin/env python3
# -*- coding: gbk -*-
'''
Created on 2018��7��31��

@author: TianHeju
'''
import numpy as np
#from FCAutoANNMakeData import FCAutoANNMakeData
from OneCNNAutoANNMakeData.makedata import makedata
class traintestset(object):
    def __init__(self,numdimension,num_train,num_test):
        self.numdimension = numdimension
        self.num_train = num_train
        self.num_test = num_test

    def buildset(self):
        #ѵ����
        train_x1 = makedata(self.numdimension, self.num_train)
        train_x, train_y = train_x1.buildnum()

        # ���Լ�
        test_x1 = makedata(self.numdimension, self.num_test)
        test_x, test_y = test_x1.buildnum()

        #�������ݼ����ļ�
        np.savez('/tmp/Fullconnectionannresult/ONECNN-'+'dimension-'+str(self.numdimension)+
                 '-dataset-'+str(self.num_train)+'X'+str(self.num_test)+'X'+str(1),
                 train_x=train_x, test_x=test_x,
                 train_y=train_y, test_y=test_y)

        #��ά������ʾ�ļ�
        # np.savetxt('train_x', train_x)
        # np.savetxt('test_x', test_x)
        # np.savetxt('train_y', train_y)
        # np.savetxt('test_y', test_y)

        #������ʾ�����
        # print(train_x.shape)
        # print(train_y.shape)
        # print(train_x[0])
        # print(train_y[0])
        #
        # print(test_x.shape)
        # print(test_x.shape)
        # print(test_x[0])
        # print(test_y[0])


