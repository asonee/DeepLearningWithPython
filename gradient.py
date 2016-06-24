# -*- coding: utf-8 -*-
"""
Created on Fri May 27 16:38:49 2016

@author: TuGuanghui

@E-mail: guanghuitu@gmail.com
实现梯度检验
"""

import numpy as np

"""
计算梯度
@param J
 代价函数
@param theta
 参数向量
@return 梯度
"""
def computeNumericGradient(J, theta):
    epsilon = 0.0001
    
    numGrad = np.zeros(theta.shape[0], dtype = np.float64) #初始化导数为0.0
    for i in range(numGrad.shape[0]):
        thetaPlusEpsilon = np.array(theta, dtype = np.float64)
        thetaPlusEpsilon[i] = theta[i] + epsilon
        thetaMinusEpsilon = np.array(theta, dtype = np.float64)
        thetaMinusEpsilon[i] = theta[i] - epsilon
        numGrad[i] = (J(thetaPlusEpsilon)[0] - J(thetaMinusEpsilon)[0]) / (2 * epsilon)
    return numGrad
 
"""
梯度检验函数
@param grad
 通过解析解计算出的梯度
@param numGrad
 通过定义计算出的梯度
"""   
def checkGradient(grad, numGrad):
    diff = np.linalg.norm(numGrad - grad)/np.linalg.norm(numGrad + grad)
    print "gradient checking result: "
    print 'the diff is: ' '%s' % diff
    print 'Norm of the difference between numerical and analytical gradient (should be < 1e-9)\n\n'
    
if __name__ == '__main__':
    
    def simpleQuadraticCostFun(theta):
        cost = theta[0] ** 2 + 3 * theta[0] * theta[1]
        
        grad = np.zeros(2)
        grad[0] = 2 * theta[0] + 3 * theta[1]
        grad[1] = 3 * theta[0]
        
        return cost, grad
        
    theta = np.array([4, 10])
    cost, grad = simpleQuadraticCostFun(theta)
    numGrad = computeNumericGradient(simpleQuadraticCostFun, theta)
    checkGradient(grad, numGrad)

