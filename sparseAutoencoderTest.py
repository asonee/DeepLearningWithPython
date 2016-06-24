# -*- coding: utf-8 -*-
"""
Created on Fri May 27 16:38:49 2016

@author: TuGuanghui

@E-mail: guanghuitu@gmail.com
"""

import numpy as np
import scipy.optimize
import loadMnist
import gradient
import sparseAutoencoder
import displayNetwork

##======================================================================
## STEP 0: 初始化超参数
visibleSize = 28 * 28 #输入层神经元的个数
hiddenSize = 196  #隐藏层神经元个数
sparsityParam = 0.1  #稀疏参数， 期望的每个隐藏层神经元的平均激活度
lambda_ = 3e-3  #权重惩罚项的权重
beta = 3 #稀疏惩罚项的权重
gradientCheck = False #是否进行导数检验

##======================================================================
## STEP 1: 加载图片
images = loadMnist.loadMnistImages(".\\dataset\\mnist\\train-images-idx3-ubyte")
patches = images[:, 0 : 100]

##======================================================================
## STEP 2: 初始化参数初值
theta = sparseAutoencoder.initialize(hiddenSize, visibleSize)

##======================================================================
## STEP 3: 梯度检验
if gradientCheck:
    cost, grad = sparseAutoencoder.sparseAutoencoderCost(theta, visibleSize, hiddenSize, lambda_, sparsityParam, beta, patches)
    J = lambda x: sparseAutoencoder.sparseAutoencoderCost(x, visibleSize, hiddenSize, lambda_, sparsityParam, beta, patches)
    numGrad = gradient.computeNumericGradient(J, theta)
    gradient.checkGradient(grad, numGrad)
     
##======================================================================
## STEP 4: 算法实现检验完成之后，对稀疏自编码器进行测试
J = lambda x: sparseAutoencoder.sparseAutoencoderCost(x, visibleSize, hiddenSize,
                                                         lambda_, sparsityParam,
                                                         beta, patches)
options_ = {'maxiter': 400, 'disp': True}
result = scipy.optimize.minimize(J, theta, method='L-BFGS-B', jac=True, options=options_)
opt_theta = result.x
print result

##======================================================================
## STEP 5: 可视化学习到的特征
W1 = opt_theta[0:hiddenSize * visibleSize].reshape(hiddenSize, visibleSize)
displayNetwork.displayNetwork(W1.T, "weights.png")


