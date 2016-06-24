# -*- coding: utf-8 -*-
"""
Created on Fri May 27 16:38:49 2016

@author: TuGuanghui

@E-mail: guanghuitu@gmail.com
实现稀疏自编码器算法
"""
import numpy as np
import scipy.optimize

"""
根据隐藏层的神经元个数和输入层神经元的个数随机初始化W
@param hiddenSize
 隐藏层神经元的个数
@param visibleSize
 输入层神经元的个数
@return 参数向量，里面包含W,b
"""
def initialize(hiddenSize, visibleSize):
    #将W随机初始化到[-r , r]区间
    r = np.sqrt(6) / np.sqrt(hiddenSize + visibleSize + 1)
    W1 = np.random.random((hiddenSize, visibleSize)) * 2 * r -r
    W2 = np.random.random((visibleSize, hiddenSize)) * 2* r - r
    #将b初始化为0.0
    b1 = np.zeros(hiddenSize, dtype = np.float64)
    b2 = np.zeros(visibleSize, dtype = np.float64)
    #将W和b放入一个向量中
    theta = np.concatenate((W1.reshape(hiddenSize * visibleSize), W2.reshape(hiddenSize * visibleSize), b1, b2))
    return theta
    
"""
计算稀疏编码损失及梯度
@param theta
 权重向量
@param visibleSize
 输入层神经元的个数
@param hiddenSize
 隐藏层神经元的个数
@param lambda_
 权重衰减系数
@param sparsity_param
 稀疏值，可以用它指定我们所需的稀疏程度 
@param beta
 稀疏值惩罚项的权重
@param data
 训练样本, n x m矩阵，每一列代表一个样本
@return (cost, grad)即(损失, 梯度)
""" 
def sparseAutoencoderCost(theta, visibleSize, hiddenSize, lambda_, sparsity_param, beta, data):
    #将theta展开为W1, W2, b1, b2
    W1 = theta[0 : hiddenSize * visibleSize].reshape((hiddenSize, visibleSize))
    W2 = theta[hiddenSize * visibleSize : 2 * hiddenSize * visibleSize].reshape((visibleSize, hiddenSize))
    b1 = theta[2 * hiddenSize * visibleSize : 2 * hiddenSize * visibleSize + hiddenSize]
    b2 = theta[2 * hiddenSize * visibleSize + hiddenSize :]
    
    m = data.shape[1] #获取样本总数
    
    #前向传播，得到 z和 a
    z2 = np.dot(W1 ,data) + np.tile(b1, (m, 1)).T
    a2 = sigmoid(z2)
    z3 = np.dot(W2, a2) + np.tile(b2, (m, 1)).T
    h = sigmoid(z3) # h也可以表示为 a3
    
    #计算隐藏单元i的平均激活值rho hat
    rhoHat = np.sum(a2, axis = 1) / m
    rho = np.tile(sparsity_param, hiddenSize)
    
    #计算cost
    cost = np.sum((data - h) ** 2 )/ (2 * m) + \
    lambda_ * ( np.sum(W1 ** 2) + np.sum(W2 ** 2)) / 2.0 + \
    beta * np.sum(KLDivergence(rho, rhoHat))
        
    #后向传播，获取delta
    spasityDelta = -( rho / rhoHat) + (1 - rho) / (1 - rhoHat)
    spasityDelta = np.tile(spasityDelta, (m, 1)).T
    delta3 = -(data - h)* sigmoidPrime(z3)
    delta2 = (np.dot(W2.T, delta3) + beta * spasityDelta) * sigmoidPrime(z2)
    
    #计算W1, W2, b1 ,b2梯度
    W1Grad = np.dot(delta2, data.T)/ m + lambda_ * W1
    b1Grad = np.sum(delta2, axis = 1) / m
    W2Grad = np.dot(delta3, a2.T) / m + lambda_ * W2
    b2Grad = np.sum(delta3, axis = 1) / m
    
    #将所有求得的导数的矩阵形式合并为一个向量的形式
    grad = np.concatenate((W1Grad.flatten(), W2Grad.flatten(), b1Grad.flatten(), b2Grad.flatten()))
    return cost, grad

"""
计算使用线性解码器的稀疏编码器的损失及梯度
@param theta
 权重向量
@param visibleSize
 输入层神经元的个数
@param hiddenSize
 隐藏层神经元的个数
@param lambda_
 权重衰减系数
@param sparsity_param
 稀疏值，可以用它指定我们所需的稀疏程度 
@param beta
 稀疏值惩罚项的权重
@param data
 训练样本, n x m矩阵，每一列代表一个样本
@return (cost, grad)即(损失, 梯度)
""" 
def sAEWithLinearDecoderCost(theta, visibleSize, hiddenSize, lambda_, sparsity_param, beta, data):
    #将theta展开为W1, W2, b1, b2
    W1 = theta[0 : hiddenSize * visibleSize].reshape((hiddenSize, visibleSize))
    W2 = theta[hiddenSize * visibleSize : 2 * hiddenSize * visibleSize].reshape((visibleSize, hiddenSize))
    b1 = theta[2 * hiddenSize * visibleSize : 2 * hiddenSize * visibleSize + hiddenSize]
    b2 = theta[2 * hiddenSize * visibleSize + hiddenSize :]
    
    m = data.shape[1] #获取样本总数
    
    #前向传播，得到 z和 a
    z2 = np.dot(W1 ,data) + np.tile(b1, (m, 1)).T
    a2 = sigmoid(z2)
    z3 = np.dot(W2, a2) + np.tile(b2, (m, 1)).T
    h = z3 # h也可以表示为 a3
    
    #计算隐藏单元i的平均激活值rho hat
    rhoHat = np.sum(a2, axis = 1) / m
    rho = np.tile(sparsity_param, hiddenSize)
    
    #计算cost
    cost = np.sum((data - h) ** 2 )/ (2 * m) + \
    lambda_ * ( np.sum(W1 ** 2) + np.sum(W2 ** 2)) / 2.0 + \
    beta * np.sum(KLDivergence(rho, rhoHat))
        
    #后向传播，获取delta
    spasityDelta = -( rho / rhoHat) + (1 - rho) / (1 - rhoHat)
    spasityDelta = np.tile(spasityDelta, (m, 1)).T
    delta3 = -(data - h)
    delta2 = (np.dot(W2.T, delta3) + beta * spasityDelta) * sigmoidPrime(z2)
    
    #计算W1, W2, b1 ,b2梯度
    W1Grad = np.dot(delta2, data.T)/ m + lambda_ * W1
    b1Grad = np.sum(delta2, axis = 1) / m
    W2Grad = np.dot(delta3, a2.T) / m + lambda_ * W2
    b2Grad = np.sum(delta3, axis = 1) / m
    
    #将所有求得的导数的矩阵形式合并为一个向量的形式
    grad = np.concatenate((W1Grad.flatten(), W2Grad.flatten(), b1Grad.flatten(), b2Grad.flatten()))
    return cost, grad


    
"""
训练稀疏编码器
@param theta
 权重向量
@param visibleSize
 输入层神经元的个数
@param hiddenSize
 隐藏层神经元的个数
@param lambda_
 权重衰减系数
@param sparsity_param
 稀疏值，可以用它指定我们所需的稀疏程度 
@param beta
 稀疏值惩罚项的权重
@param data
 训练样本，n x m矩阵，每一列代表一个样本
@param options_
 可选选项
@return 最优参数
""" 
def train(theta, visibleSize, hiddenSize, lambda_, sparsityParam, beta, data, options_ = {'maxiter': 400, 'disp': True}):
    #定义损失函数
    J = lambda x: sparseAutoencoderCost(x, visibleSize, hiddenSize, lambda_, sparsityParam, beta, data)
    #进行优化
    result = scipy.optimize.minimize(J, theta, method='L-BFGS-B', jac=True, options=options_)
    print result
    opt_theta = result.x
    return opt_theta
    
"""
训练使用线性解码器的稀疏编码器
@param theta
 权重向量
@param visibleSize
 输入层神经元的个数
@param hiddenSize
 隐藏层神经元的个数
@param lambda_
 权重衰减系数
@param sparsity_param
 稀疏值，可以用它指定我们所需的稀疏程度 
@param beta
 稀疏值惩罚项的权重
@param data
 训练样本，n x m矩阵，每一列代表一个样本
@param options_
 可选选项
@return 最优参数
""" 
def trainSAEWithLinearDecoder(theta, visibleSize, hiddenSize, lambda_, sparsityParam, beta, data, options_ = {'maxiter': 400, 'disp': True}):
    #定义损失函数
    J = lambda x: sAEWithLinearDecoderCost(x, visibleSize, hiddenSize, lambda_, sparsityParam, beta, data)
    #进行优化
    result = scipy.optimize.minimize(J, theta, method='L-BFGS-B', jac=True, options=options_)
    print result
    opt_theta = result.x
    return opt_theta

"""
将原始数据映射到新的特征空间上，即对原始数据进行编码
@param theta
 权重向量
@param visibleSize
 输入层神经元的个数
@param hiddenSize
 隐藏层神经元的个数
@param data
 训练样本，n x m矩阵，每一列代表一个样本
@return 编码后的数据
""" 
def sparseAutoencoder(theta, visibleSize, hiddenSize, data):
    #将theta展开为W1, W2, b1, b2
    W1 = theta[0 : hiddenSize * visibleSize].reshape((hiddenSize, visibleSize))
    b1 = theta[2 * hiddenSize * visibleSize : 2 * hiddenSize * visibleSize + hiddenSize]
    
    m = data.shape[1] #获取样本总数
    
    #前向传播获取激活值
    z2 = np.dot(W1, data) + np.tile(b1, (m, 1)).T
    a2 = sigmoid(z2)
    return a2
    

#计算sigmoid函数值    
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-1.0 * x))

#计算sigmoid函数的导数值
def sigmoidPrime(x):
    return sigmoid(x) * (1.0 - sigmoid(x))
    
#KL损失    
def KLDivergence(x, xHat):
    return x * np.log( x / xHat) + (1.0 - x) * np.log((1.0 - x) / (1.0 - xHat))
    
if __name__ == '__main__':
    """使用梯度检验验证sparseAutoencoder是否正确实现"""
    
    import gradient
    import loadMnist
    ## STEP 0: 初始化超参数
    visibleSize = 28 * 28 #输入层神经元的个数
    hiddenSize = 2  #隐藏层神经元个数
    sparsityParam = 0.1  #稀疏参数， 期望的每个隐藏层神经元的平均激活度
    lambda_ = 3e-3  #权重惩罚项的权重
    beta = 3 #稀疏惩罚项的权重
    gradientCheck = False #是否进行导数检验

    ## STEP 1: 加载图片
    images = loadMnist.loadMnistImages(".\\dataset\\mnist\\train-images-idx3-ubyte")
    patches = images[:, 0 : 2]
    
    ##STEP 2: 进行梯度检验
    theta = initialize(hiddenSize, visibleSize)
    #验证sparseAutoencoderCost是否实现正确
    """
    cost, grad = sparseAutoencoderCost(theta, visibleSize, hiddenSize, lambda_, sparsityParam, beta, patches)
    J = lambda x: sparseAutoencoderCost(x, visibleSize, hiddenSize, lambda_, sparsityParam, beta, patches)
    numGrad = gradient.computeNumericGradient(J, theta)
    gradient.checkGradient(grad, numGrad)
    """
    #验证sAEWithLinearDecoderCost是否实现正确
    cost, grad = sAEWithLinearDecoderCost(theta, visibleSize, hiddenSize, lambda_, sparsityParam, beta, patches)
    J = lambda x: sAEWithLinearDecoderCost(x, visibleSize, hiddenSize, lambda_, sparsityParam, beta, patches)
    numGrad = gradient.computeNumericGradient(J, theta)
    gradient.checkGradient(grad, numGrad)
    