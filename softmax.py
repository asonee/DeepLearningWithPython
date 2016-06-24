# -*- coding: utf-8 -*-
"""
Created on Wed Jun 08 20:16:33 2016

@author: TuGuanghui

@E-mail: guanghuitu@gmail.com
实现softmax分类器，需要用户指定的参数1）最大迭代次数 2）lambda权重衰减参数
"""
import numpy as np
import scipy.optimize
import scipy.sparse

"""
训练softmax分类器
@param theta
 已经展开的参数向量
@param lambda_
 权重衰减系数
@param numClasses
 类别数量
@param data
 训练数据
@param labels
 训练数据标签
@param options_
 训练需要用户指定的参数
@return 训练好的模型
"""
def buildClassifier(theta, lambda_, numClasses, data, labels, options_ = {'maxiter': 400, 'disp': True}):
    n, m = data.shape
    J = lambda x: softmaxCost(x, lambda_, numClasses, data, labels)
    result = scipy.optimize.minimize(J, theta, method='L-BFGS-B', jac=True, options=options_) #利用L-BFGS-B进行最优化
    opt_theta = result.x
    print result
    model = {}
    model["W"] = opt_theta[0: numClasses * n].reshape(numClasses, n)
    model["b"] =  opt_theta[numClasses * n : ]
    return model
    
"""
分类方法
@param model
 训练好的模型
@param inX
 需要分类的数据，每一列为一个样本
@return 类别
""" 
def classify(model ,inX):
    m = inX.shape[1] #需要分类的样本数量
    W = model["W"]
    b = model["b"]
    K, n = W.shape #K代表类别数， n代表特征数目
    z = W.dot(inX) + np.tile(b, (m, 1)).T
    predProb = np.exp(z) / np.sum(np.exp(z), axis = 0)
    return np.argmax(predProb, axis = 0)
    
"""
计算softmax代价函数的梯度和损失
@param theta
 已经展开的参数向量
@param lambda_
 权重衰减系数
@param numClasses
 类别数量
@param data
 训练数据
@param labels
 训练数据标签
@return 损失和梯度
""" 
def softmaxCost(theta, lambda_, numClasses, data, labels):
    n, m = data.shape #n代表特征的数目， m代表样本的数目
    K = numClasses #K代表类别数目
    #将theta展开成W, b
    W = theta[0 : K * n].reshape(K, n)
    b = theta[K * n :]
    #计算每个样本分别属于每个类别的归一化概率
    z = W.dot(data) + np.tile(b, (m, 1)).T #计算Wx + b
    z = z - np.max(z) #防止z过大，导致e^z上溢出
    predProbalities = np.exp(z) #计算e^(Wx +b)
    norm = np.sum(predProbalities, axis = 0) #概率规范化项
    predProbalities = predProbalities / np.tile(norm, (K, 1))
    #将标签转换成one-hot Vector形式
    indicator = scipy.sparse.csr_matrix((np.ones(m), (labels, np.array(range(m)))))
    indicator = np.array(indicator.todense())
    #计算cost
    cost = (-1.0 / m) * np.sum(indicator * np.log(predProbalities)) + 0.5 * lambda_ * np.sum(W * W)
    #计算导数
    grad4W = (-1.0 / m) * (indicator - predProbalities).dot(data.T) + lambda_ * W
    grad4b = (-1.0 / m) * np.sum((indicator - predProbalities), axis = 1)
    #将计算好的导数封装成一个向量
    grad = np.concatenate((grad4W.flatten(), grad4b))
    return cost, grad

    
    
if __name__ == '__main__':
    import loadMnist
    import gradient
    images = loadMnist.loadMnistImages(".\\dataset\\mnist\\train-images-idx3-ubyte")
    labels = loadMnist.loadMnistLabels(".\\dataset\\mnist\\train-labels-idx1-ubyte")
    lambda_ = 0.0001
    numClasses = 10
    featNum = images.shape[0]
    theta =  np.random.randn(numClasses * (featNum + 1)) #利用正态分布随机初始化W 以及 b
    isGradientCheck = False
    if isGradientCheck:
        #进行梯度检验
        imagePatches = images[:, 0: 100]
        labelsPatches = labels[0: 100]
        cost, grad = softmaxCost(theta, lambda_, numClasses, imagePatches, labelsPatches)
        J = lambda x: softmaxCost(x, lambda_, numClasses, imagePatches, labelsPatches)
        numGrad = gradient.computeNumericGradient(J, theta)
        gradient.checkGradient(grad, numGrad)
    #进行模型训练
    options_ = {'maxiter': 400, 'disp': True}
    model = buildClassifier(theta, lambda_, numClasses, images, labels, options_)
    #进行模型测试
    testImages = loadMnist.loadMnistImages(".\\dataset\\mnist\\t10k-images-idx3-ubyte")
    testLabels = loadMnist.loadMnistLabels(".\\dataset\\mnist\\t10k-labels-idx1-ubyte")
    predLabels = classify(model, testImages)
    print "accuracy: ", np.sum(predLabels == testLabels) / float(testLabels.shape[0])

