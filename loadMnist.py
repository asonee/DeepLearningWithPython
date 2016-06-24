# -*- coding: utf-8 -*-
"""
Created on Fri May 27 16:14:48 2016

@author: TuGuanghui

@E-mail: guanghuitu@gmail.com

加载mnist数据集
"""
import numpy as np

"""
加载mnist图像文件
@param fileName
 读取的图像文件的文件名
@return 图像矩阵
"""
def loadMnistImages(fileName):
    f = open(fileName, "rb") #以二进制可读形式打开文件
    
    magic = np.fromfile(f, dtype=np.dtype('>i4'), count=1)
    numImages = np.fromfile(f, dtype=np.dtype('>i4'), count=1) #读取图像个数
    numRows = np.fromfile(f, dtype=np.dtype('>i4'), count=1) #读取图像长
    numCols = np.fromfile(f, dtype=np.dtype('>i4'), count=1) #读取图像宽
    
    images = np.fromfile(f, dtype=np.uint8)
    images = images.reshape((numImages, numRows * numCols)).T #转换成矩阵形式
    images = images.astype(np.float64) / 255 #进行归一化操作，规范化到[0, 1]区间
    f.close()
    return images
    
"""
加载mnist图像标签
@param fileName
 读取的图像标签的文件名
@return 标签矩阵
"""
def loadMnistLabels(fileName):
    f = open(fileName, "rb") #以二进制可读形式打开文件
    
    magic = np.fromfile(f, dtype=np.dtype('>i4'), count=1)
    numLabels = np.fromfile(f, dtype=np.dtype('>i4'), count=1) #读取标签个数
    
    images = np.fromfile(f, dtype=np.uint8)
    f.close()
    return images
    
if __name__ == '__main__':
     images = loadMnistImages(".\\dataset\\mnist\\train-images-idx3-ubyte")
     labels = loadMnistLabels(".\\dataset\\mnist\\train-labels-idx1-ubyte")
     print images.shape
     print labels.shape
     #显示读取的图像
     import matplotlib.pyplot as plt
     import matplotlib
     plt.imshow(images[:, 2].reshape((28,28)), cmap = matplotlib.cm.gray_r) #gray: 0为黑色， 1为白色， gray_r: 0为白色， 1为黑色
     #保存读取的图像
     #plt.imsave("test.png", images[:, 10].reshape((28,28)), cmap = matplotlib.cm.gray_r)
     import displayNetwork
     displayNetwork.displayNetwork(images[:, 0:400], "./outputs/mnist.png")


