# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 11:07:38 2016

@author: TuGuanghui

@E-mail: guanghuitu@gmail.com
实现栈式自编码网络，即多层自编码网络
"""
import numpy as np
import scipy
import scipy.optimize

"""
定义网络配置的类
"""
class NetConfig:
    def __init__(self):
        self.inputSize = 0 #inputSize存放输入层神经元的个数
        self.hiddenLayerSizes = [] #hiddenLayerSizes存放各个隐藏层神经元的个数
        self.outputSize = 0 #outputSize存放输出层神经元的个数

"""
定义存放连接权的类
"""
class Layer:
    def __init__(self, num):
        self.num = num
        self.W = None #W指第num和num+1层的连接权
        self.b = None #b指第num+1层的神经元的bias

"""
将存放自编码网络(不包括softmax层)各层的连接权stack拼接成一个向量
@param stack
 各层连接权
@return 连接权向量
"""        
def stack2Params(stack):
    params = []
    for layer in stack:
        params.append(layer.W.flatten())
        params.append(layer.b.flatten())
    params = np.concatenate(params)
    return params
  
"""
将连接权向量重构为自编码网络(不包括softmax层)各层的连接权
@param params
 连接权向量
@param netConfig
 网络配置
@return stack
"""    
def params2Stack(params, netConfig):
    stack = []
    numHiddenLayer = len(netConfig.hiddenLayerSizes)
    preLayerSize = netConfig.inputSize
    start = 0
    for i in range(numHiddenLayer):
        layer = Layer(i+1)
        layer.W = params[start : (start + netConfig.hiddenLayerSizes[i] * preLayerSize)].reshape(netConfig.hiddenLayerSizes[i], preLayerSize)
        start = (start + netConfig.hiddenLayerSizes[i] * preLayerSize)
        layer.b = params[start : (start + netConfig.hiddenLayerSizes[i])]
        stack.append(layer)
        start = (start + netConfig.hiddenLayerSizes[i])
        preLayerSize = netConfig.hiddenLayerSizes[i]
    return stack
  
"""
计算多层自编码网络损失及梯度
@param theta
 权重向量
@param numClasses
 类别数量
@param netConfig
 网络配置
@param lambda_
 权重衰减系数
@param data
 训练样本, n x m矩阵，每一列代表一个样本
@param labels
 训练样本标签
@return (cost, grad)即(损失, 梯度)
"""   
def stackedAutoencoderCost(theta, numClasses, netConfig, lambda_, data, labels):
    ##Step 0: 获取数据集和神经网络相关信息
    m = data.shape[1] #m为样本的数目
    K = numClasses #K存储类别的数量
    outputSize = netConfig.outputSize
    hiddenSizes = netConfig.hiddenLayerSizes #hiddenSizes存放各个隐藏层神经元的个数
    
    ##Step 1: 从theta向量提取出相应层的连接权
    softmaxW = theta[0 : outputSize * hiddenSizes[-1]].reshape((outputSize, hiddenSizes[-1]))
    softmaxb = theta[outputSize * hiddenSizes[-1] : (outputSize * hiddenSizes[-1] + outputSize)]
    params = theta[(outputSize * hiddenSizes[-1] + outputSize) : ]
    stack = params2Stack(params, netConfig)
    
    ##Step 2: 前向传播，计算z和a
    a = [data] #存放所有自编码层的acitivation，初始化为data
    z = [np.array(0)] #存放所有自编码层的z, 初始化为空
    for layer in stack:
        currentZ = layer.W.dot(a[-1]) + np.tile(layer.b, (m, 1)).T
        z.append(currentZ)
        a.append(sigmoid(z[-1]))
    #计算softmax层,每个样本分别属于每个类别的归一化概率
    softmaxZ = softmaxW.dot(a[-1]) + np.tile(softmaxb, (m, 1)).T #计算Wx + b
    softmaxZ = softmaxZ - np.max(softmaxZ) #防止z过大，导致e^z上溢出
    softmaxA = np.exp(softmaxZ) #计算e^(Wx +b)
    norm = np.sum(softmaxA, axis = 0) #概率规范化项
    softmaxA = softmaxA / np.tile(norm, (K, 1))

    ##Step 3: 计算cost及softmax的导数
    #将标签转换成one-hot Vector形式
    indicator = scipy.sparse.csr_matrix((np.ones(m), (labels, np.array(range(m)))))
    indicator = np.array(indicator.todense())
    cost = (-1.0 / m) * np.sum(indicator * np.log(softmaxA)) + 0.5 * lambda_ * np.sum(softmaxW * softmaxW) 
    softmaxGrad4W = (-1.0 / m) * (indicator - softmaxA).dot(a[-1].T) + lambda_ * softmaxW
    softmaxGrad4b = (-1.0 / m) * np.sum((indicator - softmaxA), axis = 1)
    
    ##Step 4: 反向传播,计算delta
    softmaxGradA = -softmaxW.transpose().dot(indicator - softmaxA) #计算softmax代价函数对x的导数
    delta = [softmaxGradA * sigmoidPrime(z[-1])] #初始化自编码最后一层的delta，即这里的最后一层不是指softmax这一层
    for i in reversed(range(len(hiddenSizes) - 1)):
        currentDelta = stack[i + 1].W.T.dot(delta[0]) * sigmoidPrime(z[i + 1]) #计算其余的除第一层的delta
        delta.insert(0, currentDelta)
    
    ##Step 5: 计算自编码层连接权的导数
    stackGrad = []
    for i in range(len(hiddenSizes)):
        layer = Layer(i + 1)
        layer.W = delta[i].dot(a[i].T) / float(m)
        layer.b = np.sum(delta[i], axis=1) / float(m)
        stackGrad.append(layer)
    
    ##Step 6: 将计算好的所有导数组成一个向量
    stackGrad = stack2Params(stackGrad)
    grad = np.concatenate((softmaxGrad4W.flatten(), softmaxGrad4b.flatten(), stackGrad))
    return cost, grad
  
"""
微调多层自编码网络
@param theta
 权重向量
@param numClasses
 类别数量
@param netConfig
 网络配置
@param lambda_
 权重衰减系数
@param data
 训练样本, n x m矩阵，每一列代表一个样本
@param labels
 训练样本标签
@param options
 可选选项
@return 最优参数
"""  
def fineTuning(theta, numClasses, netConfig, lambda_, data, labels, options = {'maxiter': 400, 'disp': True} ):
    J = lambda x: stackedAutoencoderCost(x, numClasses, netConfig, lambda_, data, labels)
    result = scipy.optimize.minimize(J, theta, method='L-BFGS-B', jac=True, options=options) #利用L-BFGS-B进行最优化
    optTheta = result.x
    return optTheta
    
"""
对数据进行分类
@param theta
 权重向量
@param numClasses
 类别数量
@param netConfig
 网络配置
@return 样本标签
"""
def classify(theta, numClasses, netConfig, data):
    ##Step 0: 获取数据集和神经网络相关信息
    m = data.shape[1] #m为样本的数目
    K = numClasses #K存储类别的数量
    outputSize = netConfig.outputSize
    hiddenSizes = netConfig.hiddenLayerSizes #hiddenSizes存放各个隐藏层神经元的个数
    
    ##Step 1: 从theta向量提取出相应层的连接权
    softmaxW = theta[0 : outputSize * hiddenSizes[-1]].reshape((outputSize, hiddenSizes[-1]))
    softmaxb = theta[outputSize * hiddenSizes[-1] : (outputSize * hiddenSizes[-1] + outputSize)]
    params = theta[(outputSize * hiddenSizes[-1] + outputSize) : ]
    stack = params2Stack(params, netConfig)
    
    ##Step 2: 进行前向传播
    a = data #acitivation，初始化为data
    for layer in stack:
        currentZ = layer.W.dot(a) + np.tile(layer.b, (m, 1)).T
        a = sigmoid(currentZ)
    softmaxZ = softmaxW.dot(a) + np.tile(softmaxb, (m, 1)).T #计算Wx + b
    softmaxZ = softmaxZ - np.max(softmaxZ) #防止z过大，导致e^z上溢出
    softmaxA = np.exp(softmaxZ) #计算e^(Wx +b)
    norm = np.sum(softmaxA, axis = 0) #概率规范化项
    softmaxA = softmaxA / np.tile(norm, (K, 1))
    
    return np.argmax(softmaxA, axis = 0)

#计算sigmoid函数值    
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-1.0 * x))

#计算sigmoid函数的导数值
def sigmoidPrime(x):
    return sigmoid(x) * (1.0 - sigmoid(x))
   
if __name__ == '__main__':
    """使用梯度检验验证stack_autoencoder是否正确实现"""
    import gradient
    ## STEP 0: 配置栈式自编码神经网络
    netConfig = NetConfig()
    netConfig.inputSize = 64
    netConfig.outputSize = 4
    netConfig.hiddenLayerSizes = [36, 25]
    numClasses = 4
    ## STEP 1: 加载数据集
    data = np.random.randn(netConfig.inputSize, 10)
    labels = np.random.randint(4, size=10)
    ## STEP 2: 将各层的连接权初始化及其他参数初始化
    lambda_ = 0.01
    stack = [] #stack存储所有自编码层的链接权
    preLayerSize = netConfig.inputSize
    for i in range(len(netConfig.hiddenLayerSizes)):
        layer = Layer(i+1)
        layer.W = 0.1 * np.random.randn(netConfig.hiddenLayerSizes[i], preLayerSize)
        layer.b = np.random.randn(netConfig.hiddenLayerSizes[i])
        stack.append(layer)
        preLayerSize = netConfig.hiddenLayerSizes[i]
    softmaxTheta =  0.005 * np.random.randn((netConfig.hiddenLayerSizes[-1] + 1) * netConfig.outputSize)
    ## STEP 3: 将所有参数合并放入一个向量里面
    params = stack2Params(stack)
    theta = np.concatenate((softmaxTheta, params))
    ## STEP 4: 梯度检验
    #stackedAutoencoderCost(theta, numClasses, netConfig, lambda_, data, labels)
    cost, grad = stackedAutoencoderCost(theta, numClasses, netConfig, lambda_, data, labels)
    J = lambda x : stackedAutoencoderCost(x, numClasses, netConfig, lambda_, data, labels)
    numGrad = gradient.computeNumericGradient(J, theta)
    gradient.checkGradient(grad, numGrad)
    
    
    

    

