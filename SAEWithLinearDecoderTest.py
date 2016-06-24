# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 19:35:21 2016

@author: TuGuanghui

@E-mail: guanghuitu@gmail.com
实现对使用线性解码器的稀疏自编码器的测试
"""
import scipy
import displayNetwork
import numpy as np
import sparseAutoencoder
import cPickle
import time
import datetime

##======================================================================
## STEP 0: 初始化参数
imageChannels = 3;     #number of channels (rgb, so 3)

patchDim   = 8;          #patch dimension

visibleSize = patchDim * patchDim * imageChannels;  #number of input units 
outputSize  = visibleSize;   #number of output units
hiddenSize  = 400;           #number of hidden units 

sparsityParam = 0.035; #desired average activation of the hidden units.
lambda_ = 3e-3;         #weight decay parameter       
beta = 5;              #weight of sparsity penalty term       

epsilon = 0.1;	       #epsilon for ZCA whitening

options = {'maxiter': 400, 'disp': True} #调用LBFGS的选项

##======================================================================
## STEP 1: 加载数据集
patches = scipy.io.loadmat('dataset/stlSampledPatches.mat')['patches']
#patches = patches[:, 0 : 1000]
numPatches = patches.shape[1]   #number of patches
displayNetwork.displayColorNetwork(patches[:, 0 : 100], "./outputs/stl10SampledPatches.png")

##======================================================================
## STEP 2: 对数据集进行预处理
mean = np.mean(patches, axis = 1) #分别为每个像素块计算像素强度均值
patches = patches - np.tile(mean,(numPatches, 1)).T#将数据零均值化
cov = (patches).dot(patches.T) / numPatches #计算协方差，因为均值已经为0，所以可以这样求
u, s, v = np.linalg.svd(cov) #对协方差矩阵进行特征值分解， u为特征向量， s为排好序由大到小的特征值
ZCAWhite = u.dot(np.diag(1.0 / np.sqrt(s + epsilon)).dot(u.T))
patches = ZCAWhite.dot(patches)
displayNetwork.displayColorNetwork(patches[:, 0 : 100], "./outputs/stl10SampledPatchesWithZCA.png")

##======================================================================
## STEP 3: 学习特征
startTime = time.time()
theta = sparseAutoencoder.initialize(hiddenSize, visibleSize)
optTheta = sparseAutoencoder.trainSAEWithLinearDecoder(theta, visibleSize, hiddenSize, lambda_, sparsityParam, beta, patches, options)
#保存学习到的特征和预处理的参数
f = open('stl10PatchFetures.pickle', 'wb') 
cPickle.dump(optTheta, f)
cPickle.dump(ZCAWhite, f)
cPickle.dump(mean, f)
print('Saved successfully')
endTime = time.time()
print "feature learning time elapsed: " + str(datetime.timedelta(seconds = endTime - startTime))

##======================================================================
## STEP 4: 可视化学习到的特征
W1 = optTheta[0:hiddenSize * visibleSize].reshape(hiddenSize, visibleSize)
b1 = optTheta[2 * hiddenSize * visibleSize : 2 * hiddenSize * visibleSize + hiddenSize]
displayNetwork.displayColorNetwork( W1.dot(ZCAWhite).T, "./outputs/stl10PatchFetures.png")



