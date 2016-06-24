# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 09:41:58 2016

@author: TuGuanghui

@E-mail: guanghuitu@gmail.com
对stackedAutoencoder在Mnist数据集上进行测试，当程序跑完后，结果为：
before fine tuning's accuracy:  0.8836
after fine tuning's accuracy:  0.9772
"""
import loadMnist
import stackedAutoencoder
import sparseAutoencoder
import numpy as np
import softmax
import cPickle
import displayNetwork

##======================================================================
## STEP 0: 配置栈式自编码神经网络
netConfig = stackedAutoencoder.NetConfig()
netConfig.inputSize = 28 * 28
netConfig.outputSize = 10
netConfig.hiddenLayerSizes = [200, 200]
numClasses = 10

##======================================================================
## STEP 1: 配置稀疏自编码器参数及softmax参数
#配置稀疏自编码器参数，预训练的每个自编码器参数都一样
lambda4AE = 3e-3
beta = 3
sparsityParam = 0.1
options4AE = {'maxiter': 400, 'disp': True}
#配置softmax参数
lambda4SM = 3e-3
options4SM = {'maxiter': 400, 'disp': True}
#配置stacked autoencoder参数
options4SAE = {'maxiter': 400, 'disp': True}

##======================================================================
## STEP 2: 加载数据集
trainImages = loadMnist.loadMnistImages(".\\dataset\\mnist\\train-images-idx3-ubyte")
trainLabels = loadMnist.loadMnistLabels(".\\dataset\\mnist\\train-labels-idx1-ubyte")
#trainImages = trainImages[:, 0 : 50]
#trainLabels = trainLabels[0 : 50]
##======================================================================
## STEP 3: 训练第一个稀疏自编码器
sae1TrainData = trainImages
sae1InputSize = netConfig.inputSize
sae1HiddenSize = netConfig.hiddenLayerSizes[0]
sae1Theta = sparseAutoencoder.initialize(sae1HiddenSize, sae1InputSize)
sae1OptTheta = sparseAutoencoder.train(sae1Theta, sae1InputSize, sae1HiddenSize, lambda4AE, sparsityParam, beta, sae1TrainData, options4AE)
#保存训练得到的参数
with open('sae1OptTheta.pickle', 'wb') as f:
    cPickle.dump(sae1OptTheta, f)
print('Saved successfully')

##======================================================================
## STEP 4: 训练第二个稀疏自编码器
sae2TrainData = sparseAutoencoder.sparseAutoencoder(sae1OptTheta, sae1InputSize, sae1HiddenSize, sae1TrainData)
sae2InputSize = sae1HiddenSize
sae2HiddenSize = netConfig.hiddenLayerSizes[1]
sae2Theta = sparseAutoencoder.initialize(sae2HiddenSize, sae2InputSize)
sae2OptTheta = sparseAutoencoder.train(sae2Theta, sae2InputSize, sae2HiddenSize, lambda4AE, sparsityParam, beta, sae2TrainData, options4AE)
#保存训练得到的参数
with open('sae2OptTheta.pickle', 'wb') as f:
    cPickle.dump(sae2OptTheta, f)
print('Saved successfully')

##======================================================================
## STEP 5: 训练softmax
smTrainData = sparseAutoencoder.sparseAutoencoder(sae2OptTheta, sae2InputSize, sae2HiddenSize, sae2TrainData)
smTheta =  np.random.randn(numClasses * (sae2HiddenSize + 1)) #利用正态分布随机初始化W 以及 b
smModel = softmax.buildClassifier(smTheta, lambda4SM, numClasses, smTrainData, trainLabels, options4SM)
smOptW = smModel["W"]
smOptB = smModel["b"]
#保存训练得到的模型
with open('smModel.pickle', 'wb') as f:
    cPickle.dump(smModel, f)
print('Saved successfully')
##======================================================================
## STEP 5: 对整个网络进行微调, fine-tuning
layer1 = stackedAutoencoder.Layer(1)
layer1.W = sae1OptTheta[0 : sae1HiddenSize * sae1InputSize].reshape(sae1HiddenSize, sae1InputSize)
layer1.b = sae1OptTheta[2 * sae1HiddenSize * sae1InputSize : 2 * sae1HiddenSize * sae1InputSize + sae1HiddenSize]
layer2 = stackedAutoencoder.Layer(2)
layer2.W = sae2OptTheta[0 : sae2HiddenSize * sae2InputSize].reshape(sae2HiddenSize, sae2InputSize)
layer2.b = sae2OptTheta[2 * sae2HiddenSize * sae2InputSize : 2 * sae2HiddenSize * sae2InputSize + sae2HiddenSize]
stack = [layer1, layer2]
#分别可视化第一个SparseAutoencoder和第二个SparseAutoencoder学习到的特征
displayNetwork.displayNetwork(layer1.W.T, "sae1.png")
displayNetwork.displayNetwork(layer2.W.dot(layer1.W).T, "sae2.png")

smOptTheta = np.concatenate((smOptW.flatten(), smOptB))

saeTheta = np.concatenate((smOptTheta, stackedAutoencoder.stack2Params(stack)))

saeOptTheta = stackedAutoencoder.fineTuning(saeTheta, numClasses, netConfig, lambda4SM, trainImages, trainLabels, options4SAE )

##======================================================================
## STEP 6: 测试稀疏自编码
testImages = loadMnist.loadMnistImages(".\\dataset\\mnist\\t10k-images-idx3-ubyte")
testLabels = loadMnist.loadMnistLabels(".\\dataset\\mnist\\t10k-labels-idx1-ubyte")

predLabels = stackedAutoencoder.classify(saeTheta, numClasses, netConfig, testImages)
print "before fine tuning's accuracy: ", np.sum(predLabels == testLabels) / float(testLabels.shape[0])

predLabels = stackedAutoencoder.classify(saeOptTheta, numClasses, netConfig, testImages)
print "after fine tuning's accuracy: ", np.sum(predLabels == testLabels) / float(testLabels.shape[0])