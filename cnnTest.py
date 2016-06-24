# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 20:49:59 2016

@author: TuGuanghui

@E-mail: guanghuitu@gmail.com
"""
import cPickle
import scipy.io
import numpy as np
import displayNetwork
import cnn
import softmax
import time
import datetime

##======================================================================
## STEP 0: 初始化参数
imageDim = 64;         #image的长（=宽）
imageChannels = 3;     #图像的通道数

patchDim = 8;          #patch的长（=宽）

visibleSize = patchDim * patchDim * imageChannels;  # number of input units 
outputSize = visibleSize;   # number of output units
hiddenSize = 400;           # number of hidden units 

poolDim = 19;          # dimension of pooling region

numClasses  = 4
softmaxLambda = 1e-4
softmaxOptions = {'maxiter': 200, 'disp': True} #softmax调用LBFGS的选项

##======================================================================
## STEP 1: 加载并展示由sparseAutoencoder学习得到的特征
f =  open('stl10PatchFetures.pickle', 'rb')
optTheta = cPickle.load(f)
zcaWhite = cPickle.load(f)
patchMean = cPickle.load(f)
f.close()
#展示sparseAutoencoder学习得到的特征
W1 = optTheta[0:hiddenSize * visibleSize].reshape(hiddenSize, visibleSize)
b1 = optTheta[2 * hiddenSize * visibleSize : 2 * hiddenSize * visibleSize + hiddenSize]

displayNetwork.displayColorNetwork( (W1.dot(zcaWhite) + patchMean).T, "./outputs/stl10PatchFeturesMean.png")

##======================================================================
## STEP 2: 加载数据集并展示数据集
print "step2"
stlTrain = scipy.io.loadmat('dataset/stlTrainSubset.mat')
trainImages = stlTrain['trainImages']
trainLables = stlTrain['trainLabels']
#trainImages = trainImages[:, :, :, 0:50]
#trainLables = trainLables[0:50, :]
numTrainImages = trainImages.shape[-1]

stlTest = scipy.io.loadmat('dataset/stlTestSubset.mat')
testImages = stlTest['testImages']
testLabels = stlTest['testLabels']
#testImages = testImages[:, :, :, 0:50]
#testLabels = testLabels[0:50, :]
numTestImages = testImages.shape[-1]
#展示一部分训练集
#patches = np.transpose(trainImages[:, :, :, 0 : 100], axes=[2, 0, 1, 3])
#patches = patches.reshape((-1 ,100))
#displayNetwork.displayColorNetwork(patches, ".\\outputs\\stlSubset.png")
print "step2"
##======================================================================
## STEP 3: 对数据集进行卷积和池化操作，为了避免内存溢出，每次仅适用50特征进行卷积和池化

print "start conv and pool..."
startTime = time.time()
stepSize = 20 #每次仅适用20特征进行卷积和池化
assert hiddenSize % stepSize == 0, "stepSize should divide hiddenSize"
print "start conv and pool..."
#初始化卷积池化后的数据集
pooledFeaturesTrain = np.zeros((hiddenSize, numTrainImages, int(np.floor((imageDim - patchDim + 1) / poolDim)), int(np.floor((imageDim - patchDim + 1) / poolDim))))
pooledFeaturesTest = np.zeros((hiddenSize, numTestImages, int(np.floor((imageDim - patchDim + 1) / poolDim)), int(np.floor((imageDim - patchDim + 1) / poolDim))))
print "start conv and pool..."
for covPart in range(hiddenSize / stepSize):
    print "please"
    featureStart = covPart * stepSize
    featureEnd = (covPart + 1) * stepSize
    print "setp", covPart, " features",  featureStart, "to", featureEnd
    
    Wt = W1[featureStart : featureEnd, :]
    bt = b1[featureStart : featureEnd]
    
    #对训练集进行卷积和池化
    tempConvolvedFeats = cnn.convolve(trainImages, Wt, bt, zcaWhite, patchMean)
    tempPooledFeats = cnn.pool(poolDim, tempConvolvedFeats)
    pooledFeaturesTrain[featureStart : featureEnd, :, :, :] = tempPooledFeats
    #对测试集进行卷积和池化
    tempConvolvedFeats = cnn.convolve(testImages, Wt, bt, zcaWhite, patchMean)
    tempPooledFeats = cnn.pool(poolDim, tempConvolvedFeats)
    pooledFeaturesTest[featureStart : featureEnd, :, :, :]  = tempPooledFeats
    
endTime = time.time()
print "convoluation and pooling time elapsed: " + str(datetime.timedelta(seconds = endTime - startTime))    
#将池化好的训练集和测试集进行保存
f = open('covAndPooledstlSubset.pickle', 'wb') 
cPickle.dump(pooledFeaturesTrain, f)
cPickle.dump(pooledFeaturesTest, f)
f.close()

##======================================================================
## STEP 4: 利用卷积和池化好的训练集对softmax进行训练
f = open('covAndPooledstlSubset.pickle', 'rb') 
#f = open("F:\\machine learning\\deep learning\\UFLDLTutorial\\code\\code1\\cnn_pooled_features.pickle", "rb")
pooledFeaturesTrain = cPickle.load(f)
pooledFeaturesTest = cPickle.load(f)
f.close()
softmaxTrainData = np.transpose(pooledFeaturesTrain, axes = [0, 2, 3, 1]).reshape((-1, numTrainImages))
softmaxTrainLabels = trainLables.flatten() - 1
theta =  np.random.randn(numClasses * (softmaxTrainData.shape[0] + 1)) #利用正态分布随机初始化W 以及 b

softmaxModel = softmax.buildClassifier(theta, softmaxLambda, numClasses, softmaxTrainData, softmaxTrainLabels, softmaxOptions)

##======================================================================
## STEP 4: 在卷积和池化好的测试机上对softmax进行测试
softmaxTestData = np.transpose(pooledFeaturesTest, axes = [0, 2, 3, 1]).reshape((-1, numTestImages))
softmaxTestLabels = testLabels.flatten() - 1
predLables = softmax.classify(softmaxModel, softmaxTestData)   
print "Accuracy: ",  np.sum(softmaxTestLabels == predLables) / float(numTestImages)
