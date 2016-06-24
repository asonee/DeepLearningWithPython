# -*- coding: utf-8 -*-
"""
Created on Thu Jun 02 21:03:26 2016

@author: TuGuanghui

@E-mail: guanghuitu@gmail.com
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import PIL

"""
可视化神经网络所学到的特征或者是可视化图像
@param A
 所学到的特征，A的每一列为对应于一个神经元学习到的一个新的基或者是一个图像
@param fileName
 可视化得到的图像的保存路径
"""
def displayNetwork(A, fileName):
    A = A - np.mean(A, axis = 0) #对A矩阵进行放缩，零均值化
    row, col = A.shape
    sz = np.sqrt(row) #每一个基的图像的大小，即sz * sz,小图像
    n = np.ceil(np.sqrt(col)) #一副大的图像(即所有的基的图像都放在里面)中一列放小图像的数目
    m = np.ceil(col / n) #一副大的图像(即所有的基的图像都放在里面)中一行放小图像的数目
    buf = 1 #小幅图像之间的间距
    #初始化image
    image = np.ones(shape=(buf + m * (sz + buf), buf + n * (sz + buf)))
    k = 0 #指示A的第k列
    for i in range(int(m)):#大图像的每一行
        for j in range(int(n)): #大图像的每一列
        #将小图像填入其中
            if k >= col:
                continue

            clim = np.max(np.abs(A[:, k]))
            #clim = np.linalg.norm(A[:, k]) #在UFLDL教程中"可视化自编码器训练结果"这一章中，应该clim是这样的
            image[buf + i * (sz + buf):buf + i * (sz + buf) + sz, buf + j * (sz + buf):buf + j * (sz + buf) + sz] = \
            A[:, k].reshape(sz, sz) / clim
            k += 1
    plt.imsave(fileName, image, cmap=matplotlib.cm.gray)

"""
可视化RGB图像
@param A
 每一列为一个图像
@param fileName
 可视化得到的图像的保存路径
"""    
def displayColorNetwork(A, fileName):
    if np.min(A) >= 0:
        A = A - np.mean(A) #放缩

    cols = np.round(np.sqrt(A.shape[1])) #大幅图像中每一列放的图像数目

    channel_size = A.shape[0] / 3 #每一个通道像素的总和
    dim = np.sqrt(channel_size) #小图像的长和宽的大小，或者说是像素的个数
    dimp = dim + 1 #dim padding，dim + 小幅图像之间的间距,间距为1
    rows = np.ceil(A.shape[1] / cols) #大幅图像中每一行放的图像数目

    B = A[0:channel_size, :] #B保存所有图像的第一个通道
    C = A[channel_size:2 * channel_size, :] #C保存所有图像的第一个通道
    D = A[2 * channel_size:3 * channel_size, :] #D保存所有图像的第二个通道

    B = B / np.max(np.abs(B)) #放缩
    C = C / np.max(np.abs(C))
    D = D / np.max(np.abs(D))

    #初始化大图像
    image = np.ones(shape=(dim * rows + rows - 1, dim * cols + cols - 1, 3))

    for i in range(int(rows)): #大图像的每一行
        for j in range(int(cols)): #大图像的每一列
            if (i * cols + j + 1) > B.shape[1]: 
                break
            #将小图像放入大图像当中
            image[i * dimp:i * dimp + dim, j * dimp:j * dimp + dim, 0] = B[:, i * cols + j].reshape(dim, dim)
            image[i * dimp:i * dimp + dim, j * dimp:j * dimp + dim, 1] = C[:, i * cols + j].reshape(dim, dim)
            image[i * dimp:i * dimp + dim, j * dimp:j * dimp + dim, 2] = D[:, i * cols + j].reshape(dim, dim)

    image = (image + 1) / 2 #使图像平稳过渡

    PIL.Image.fromarray(np.uint8(image * 255), 'RGB').save(fileName) #保存图像

