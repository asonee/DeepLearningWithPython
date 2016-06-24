# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 15:05:16 2016

@author: TuGuanghui

@E-mail: guanghuitu@gmail.com
实现卷积神经网络
"""
import numpy as np
import scipy.signal
import scipy.io

"""
卷积函数
@param images
 待卷积的图像, images(imageRow, imageCol, channel, image number)
@param W
 稀疏自编码器的连接权,每一行为一个特征
@param b
 稀疏自编码器的偏置
@param zcaWhite
 预处理中zca白化的矩阵
@param patchMean
 预处理中patch每个像素的均值
@return 已被卷积的图像
"""
def convolve(images, W, b, zcaWhite, patchMean):
    numImages = images.shape[3] #图像的数量
    imageDim = images.shape[0] #image的长（=宽）
    imageChannels = images.shape[2] #图像的通道数
    numFeatures = W.shape[0] #特征的个数
    patchDim = np.sqrt(W.shape[1] / imageChannels) #patch或者feature的长(=宽)
    patchSize = patchDim * patchDim

    #将预处理(即零均值化和白化)考虑进去
    W = W.dot(zcaWhite)
    b = b - W.dot(patchMean)    
    
    convolvedFeatures = np.zeros((numFeatures, numImages, imageDim - patchDim + 1, imageDim - patchDim + 1)) #存放被卷积后的图像
    
    for i in range(numImages): #对于每一张图象
        for j in range(numFeatures):  #对于每一个sparseAutoencoder学习到的特征
            convolvedIm = np.zeros((imageDim - patchDim + 1, imageDim - patchDim + 1))
            for k in range(imageChannels): #对于图象的每一个通道
                feature = W[j, k * patchSize : (k + 1) * patchSize].reshape((patchDim, patchDim)) #获取相应的特征
                feature = np.flipud(np.fliplr(feature)) #将图片flip，为了能够使用convolve2d()函数，在此函数中，会对feature再次flip
                im = images[:, :, k, i] #获取图片
                # The convolved feature is the sum of the convolved values for all channels
                convolvedIm = convolvedIm + scipy.signal.convolve2d(im, feature, mode = "valid")
                scipy.signal.convolve2d
            convolvedIm = sigmoid(convolvedIm + b[j]) #应用sigmoid得到激活值
            convolvedFeatures[j, i, :, :] = convolvedIm
                
    return convolvedFeatures

"""
均值池化函数
@param poolDim
 池化区域的长(=宽)
@param convolvedFeatures
 已被卷积的特征，convolvedFeatures(featureNum, imageNum, imageRow, imageCol)
@return 池化的特征， pooledFeatures(featureNum, imageNum, poolRow, poolCol)
"""    
def pool(poolDim, convolvedFeatures):
    numImages = convolvedFeatures.shape[1]
    numFeatures = convolvedFeatures.shape[0] 
    convolvedDim = convolvedFeatures.shape[2]
    
    pooledFeatures = np.zeros((numFeatures, numImages, int(np.floor(convolvedDim / poolDim)), int(np.floor(convolvedDim / poolDim)))) #初始化
    
    for i in range(int(np.floor(convolvedDim / poolDim))):
        for j in range(int(np.floor(convolvedDim / poolDim))):
            pool = convolvedFeatures[:, :, i * poolDim : (i + 1) * poolDim, j * poolDim : (j + 1) * poolDim] #得到待池化的区域
            pooledFeatures[:, :, i, j] = np.mean(np.mean(pool, axis = 2), axis = 2) #进行均值池化
    return pooledFeatures

#计算sigmoid函数值       
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
    
if __name__ == '__main__':
    """验证卷积和池化是否正确实现"""
    import cPickle
    import sparseAutoencoder
    import sys
    
    ##======================================================================
    ## 验证卷积是否正确实现
    ## STEP 0: 初始化参数
    imageChannels = 3;     #number of channels (rgb, so 3)
    
    hiddenSize  = 400;           #number of hidden units 
    sparsityParam = 0.035; #desired average activation of the hidden units.
    lambda_ = 3e-3;         #weight decay parameter       
    beta = 5;              #weight of sparsity penalty term       
    epsilon = 0.1;	       #epsilon for ZCA whitening
    ## STEP 1: 加载数据集
    patches = scipy.io.loadmat('dataset/stlSampledPatches.mat')['patches']
    #patches = patches[:, 0 : 1000]
    numPatches = patches.shape[1]   #number of patches
    patchDim = np.sqrt(patches.shape[0] / 3)
    visibleSize = patchDim * patchDim * imageChannels;  #number of input units 
    outputSize  = visibleSize;   #number of output units
    ## STEP 2: 对数据集进行预处理
    f = open('stl10PatchFetures.pickle', 'rb')
    optTheta = cPickle.load(f)
    zcaWhite = cPickle.load(f)
    patchMean = cPickle.load(f)
    ## STEP 3: 对数据集进行卷积
    W = optTheta[0:hiddenSize * visibleSize].reshape(hiddenSize, visibleSize)
    b = optTheta[2 * hiddenSize * visibleSize:2 * hiddenSize * visibleSize + hiddenSize]
    
    stl_train = scipy.io.loadmat('dataset/stlTrainSubset.mat')
    trainImages = stl_train['trainImages']
    imageDim = trainImages.shape[0]
    train_labels = stl_train['trainLabels']
    num_train_images = stl_train['numTrainImages'][0][0]
    
    ## Use only the first 8 images for testing
    conv_images = trainImages[:, :, :, 0:8]
    
    convolvedFeatures = convolve(conv_images, W, b, zcaWhite, patchMean)
    #验证是否和sparseAutoencoder的结果一样
    # For 1000 random points
    for i in range(1000):
        feature_num = np.random.randint(0, hiddenSize)
        image_num = np.random.randint(0, 8)
        image_row = np.random.randint(0, imageDim - patchDim + 1)
        image_col = np.random.randint(0, imageDim - patchDim + 1)
    
        patch = conv_images[image_row:image_row + patchDim, image_col:image_col + patchDim, :, image_num] #获得8x8x3的patch
    
        patch = np.concatenate((patch[:, :, 0].flatten(), patch[:, :, 1].flatten(), patch[:, :, 2].flatten())) #将8x8x3的patch铺平
        patch = np.reshape(patch, (patch.size, 1)) #将patch变成列向量
        patch = patch - np.tile(patchMean, (patch.shape[1], 1)).transpose() #将patch零均值化
        patch = zcaWhite.dot(patch) #将patch白化
    
        features = sparseAutoencoder.sparseAutoencoder(optTheta, visibleSize, hiddenSize, patch)
    
        if abs(features[feature_num, 0] - convolvedFeatures[feature_num, image_num, image_row, image_col]) > 1e-9:
            print 'Convolved feature does not match activation from autoencoder'
            print 'Feature Number      :', feature_num
            print 'Image Number        :', image_num
            print 'Image Row           :', image_row
            print 'Image Column        :', image_col
            print 'Convolved feature   :', convolvedFeatures[feature_num, image_num, image_row, image_col]
            print 'Sparse AE feature   :', features[feature_num, 0]
            sys.exit("Convolved feature does not match activation from autoencoder. Exiting...")
    
    print 'Congratulations! Your convolution code passed the test.'
    
    ##======================================================================
    ## 验证池化是否正确实现
    test_matrix = np.arange(64).reshape(8, 8)
    expected_matrix = np.array([[np.mean(test_matrix[0:4, 0:4]), np.mean(test_matrix[0:4, 4:8])],
                                [np.mean(test_matrix[4:8, 0:4]), np.mean(test_matrix[4:8, 4:8])]])
    
    test_matrix = np.reshape(test_matrix, (1, 1, 8, 8))
    
    pooled_features = pool(4, test_matrix)
    
    if not (pooled_features == expected_matrix).all():
        print "Pooling incorrect"
        print "Expected matrix"
        print expected_matrix
        print "Got"
        print pooled_features
    
    print 'Congratulations! Your pooling code passed the test.'
        

