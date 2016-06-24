# -*- coding: utf-8 -*-
"""
Created on Wed Jun 08 16:01:22 2016

@author: TuGuanghui

@E-mail: guanghuitu@gmail.com
实现对图片的白化处理
"""
import numpy as np
import random

"""
pcaWhitening方法
@param data
 需要使用pca白化的数据,每一列为一张图片
@param retainVarPercent
 需要保留的方差的百分比
@param epsilon
 正则化值，为了防止白化时，除以特征值当特征值很小时，发生上溢出或者下溢出
@return 已被白化的数据或者是降维后还原的数据
"""
def pcaWhitening(data, retainVarPercent, epsilon):
    n, m = data.shape
    mean = np.mean(data, axis = 1) #分别为每个像素块计算像素强度均值
    data = data - np.tile(mean,(m, 1)).T#将数据零均值化
    cov = (data).dot(data.T) / m #计算协方差，因为均值已经为0，所以可以这样求
    u, s, v = np.linalg.svd(cov) #对协方差矩阵进行特征值分解， u为特征向量， s为排好序由大到小的特征值
    if retainVarPercent == 1.0:
        dataRot = (u.T).dot(data)
        pcaWhiteningData = np.diag(1.0 / np.sqrt(s + epsilon)).dot(dataRot)  #将旋转后的数据进行白化 
        return pcaWhiteningData
    else:
        #不进行白化，仅仅降维
        k = 0 #选取的特征的数目
        for i in range(n):
            if (np.sum(s[0:(i+1)]) / np.sum(s)) >= retainVarPercent:
                k = (i + 1)
                break
        print 'Optimal k to retain '+str(retainVarPercent) +' variance is:', k
        dataRot = (u[:, 0: k].T).dot(data)
        dataRotHat = u[:, 0: k].dot(dataRot) #将降维后的数据还原，会有一部分损失
        return dataRotHat #返回还原后的数据
    
"""
zczWhitening方法
@param data
 需要使用zca白化的数据,每一列为一张图片
@param epsilon
 正则化值，为了防止白化时，除以特征值当特征值很小时，发生上溢出或者下溢出
@return 已被zca白化的数据
"""  
def zcaWhitening(data, epsilon):
    n, m = data.shape
    mean = np.mean(data, axis = 1) #分别为每个像素块计算像素强度均值
    data = data - np.tile(mean,(m, 1)).T#将数据零均值化
    cov = (data).dot(data.T) / m #计算协方差，因为均值已经为0，所以可以这样求
    u, s, v = np.linalg.svd(cov) #对协方差矩阵进行特征值分解， u为特征向量， s为排好序由大到小的特征值
    dataRot = (u.T).dot(data)
    #将旋转后的数据进行PCA白化
    pcaWhiteningData = np.diag(1.0 / np.sqrt(s + epsilon)).dot(dataRot)
     #将数据ZCA白化
    zcaWhiteningData = u.dot(pcaWhiteningData)
    return zcaWhiteningData
    
if __name__ == '__main__':
    import loadSampleImages
    import displayNetwork
    patches = loadSampleImages.loadIMAGES_RAW()
    num_samples = patches.shape[1]
    random_sel = random.sample(range(num_samples), 400) #随机选400个patth
    displayNetwork.displayNetwork(patches[:, random_sel], "./outputs/01rawdata.png")
    pcaWhiteningData = pcaWhitening(patches[:, random_sel], 1.0, 0.1)
    displayNetwork.displayNetwork(pcaWhiteningData, "./outputs/02pcaWhiteningdata1.0.png")
    displayNetwork.displayNetwork(pcaWhitening(patches[:, random_sel], 0.99, 0.1), "./outputs/02pcaVarianceRetain0.99.png")
    displayNetwork.displayNetwork(pcaWhitening(patches[:, random_sel], 0.90, 0.1), "./outputs/02pcaVarianceRetain0.9.png")
    zcaWhiteningData = zcaWhitening(patches[:, random_sel], 0.1)
    displayNetwork.displayNetwork(zcaWhiteningData, "./outputs/03zcaWhiteningData.png")
    
    
    
    

