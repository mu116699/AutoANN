#!user/bin/env python3
# -*- coding: gbk -*-
'''
Created on 2018年7月27日

@author: TianHeju
'''
import numpy as np
import math

class makedata(object):
    def __init__(self,numdimension,numnumber):
        #输入的参数为：输入数据的维度、产生数据的数量
        self.numdimension = numdimension
        self.numnumber = numnumber

    def buildnum(self):
        # make data for train data
        # make data for train data
        # make data for train data
        x1 = 35 * np.random.random((self.numnumber, self.numdimension))
        x1 = np.random.random((self.numnumber, self.numdimension))
        # 归一化

        xx = x1 / 35
        xx = x1
        # yy = xx.sum(axis=1)
        # yy =  -20*exp(-0.02*sqrt((xx[0]*xx[0]+xx[1]*xx[1])/2))-exp(0.5*(cos(2*pi*xx[0])+cos(2*pi*xx[1])))+20+exp(1);

        # 计算 20*math.exp(-0.02*math.sqrt((x[0]*x[0]+x[1]*x[1])/2))
        x = x1.transpose()
        t = 0
        for i in range(0, self.numdimension):
            t = t + x[i] * x[i]
        # print(x1[0])
        # print(t[0])
        t = np.array(t)

        # t = [math.sqrt(x) for x in t]
        t = [20 * (math.exp(-0.02 * (math.sqrt(x)))) for x in t]
        t = np.reshape(t, -1)

        # 计算 math.cos(2*pi*x[1])
        s = 0
        for i in range(0, self.numdimension):
            t1 = x[i]
            t1 = np.array(t1)
            t1 = [math.cos(2 * math.pi * x1) for x1 in t1]
            t1 = np.reshape(t1, -1)
            s = s + t1
        # print(x1[0])
        # print(s[0])

        # 计算 math.exp(0.5*(math.cos(2*pi*x[0])+math.cos(2*pi*x[1])))
        s = np.array(s)
        s = [math.exp(x1 / self.numdimension) for x1 in s]
        s = np.reshape(s, -1)
        # print(s[0])
        # yy = 20*math.exp(-0.02*math.sqrt((x[0]*x[0]+x[1]*x[1])/2))+math.exp(0.5*(math.cos(2*pi*x[0])+math.cos(2*pi*x[1])))
        y1 = -t - s + 20 + math.exp(1)
        y1min, y1max = y1.min(), y1.max()
        yy = (y1 - y1min) / (y1max - y1min)
        return xx,yy





