# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 09:34:41 2016

@author: TuGuanghui

@E-mail: guanghuitu@gmail.com
实现一种无监督的特征学习：自我学习,具体为利用稀疏自编码器来进行特征的学习，将学习到的特征输入到softmax进行分类
"""
import loadMnist
import numpy as np
import sparseAutoencoder
import displayNetwork
import softmax

##======================================================================
## STEP 0: 初始化稀疏自编码器参数
visibleSize = 28 * 28 #输入层神经元的个数
hiddenSize = 196 #隐藏层神经元个数
sparsityParam = 0.1  #期望的每个隐藏单元的平均激活度
lambda_ = 3e-3  #权重衰减系数
beta = 3  # 稀疏惩罚项的权重
numClasses = 5 #类别数量

## ======================================================================
#  STEP 1: 加载数据集
images = loadMnist.loadMnistImages(".\\dataset\\mnist\\train-images-idx3-ubyte")
labels = loadMnist.loadMnistLabels(".\\dataset\\mnist\\train-labels-idx1-ubyte")
#images = images[:, 0 : 100]
#labels = labels[0 : 100]
#利用5~9的样本来进行特征的学习
unlabeledIndex = np.argwhere(labels >= 5).flatten()
unlabeledData = images[:, unlabeledIndex]
#利用0~4的样本作为softmax的训练样本和测试样本，这些样本在使用之前都会转换到5~9学习到的特征上
labeledIndex = np.argwhere(labels < 5).flatten()
totalLabeledSamples = labeledIndex.shape[0]
#一半做训练样本
trainData = images[:, labeledIndex[0: totalLabeledSamples / 2]] 
trainLabel = labels[labeledIndex[0: totalLabeledSamples / 2]]
#一半做测试样本
testData = images[:, labeledIndex[totalLabeledSamples / 2 : ]]
testLabel= labels[labeledIndex[totalLabeledSamples / 2 : ]]

## ======================================================================
#  STEP 2: 利用无标签的样本训练稀疏自编码器，并将学习到的特征可视化
#训练稀疏自编码器
theta = sparseAutoencoder.initialize(hiddenSize, visibleSize) #初始化参数
options = {'maxiter': 400, 'disp': True} #设置最优化方法的参数
optTheta = sparseAutoencoder.train(theta, visibleSize, hiddenSize, lambda_, sparsityParam, beta, unlabeledData, options)
#将训练得到的特征可视化
W1 = optTheta[0:hiddenSize * visibleSize].reshape(hiddenSize, visibleSize).transpose()
displayNetwork.displayNetwork(W1, "stlFeats.png")

## ======================================================================
#  STEP 3: 利用学习到的特征，对有标签的训练和测试样本进行编码
encodedTrainData = sparseAutoencoder.sparseAutoencoder(optTheta, visibleSize, hiddenSize, trainData)
encodedTestData = sparseAutoencoder.sparseAutoencoder(optTheta, visibleSize, hiddenSize, testData)

## ======================================================================
#  STEP 4: 训练softmax分类器
theta =  np.random.randn(numClasses * (hiddenSize + 1)) #利用正态分布随机初始化W 以及 b
lambda_ =  1e-4 
options_ = {'maxiter': 400, 'disp': True}
model = softmax.buildClassifier(theta, lambda_, numClasses, encodedTrainData, trainLabel, options_)

## ======================================================================
#  STEP 5: 测试训练好的softmax分类器
wrong = 0
correct = 0
for i in range(encodedTestData.shape[1]):
    trueLabel = testLabel[i]
    preditLabel = softmax.classify(model, encodedTestData[:, i].reshape((encodedTestData.shape[0], 1)))
    if trueLabel == preditLabel:
        correct = correct + 1
    else:
        wrong = wrong + 1
print "accuracy: " + str(correct / float(wrong + correct))


