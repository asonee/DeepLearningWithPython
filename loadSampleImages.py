# -*- coding: utf-8 -*-
"""
Created on Wed Jun 08 16:05:24 2016

@author: TuGuanghui

@E-mail: guanghuitu@gmail.com
实现对样本的加载
"""
import scipy.io
import random
import numpy as np

"""
加载样本
@return 样本集
"""    
def loadIMAGES_RAW():
    image_data = scipy.io.loadmat('dataset/IMAGES_RAW.mat')['IMAGESr']

    patch_size = 12
    num_patches = 10000
    num_images = image_data.shape[2]
    image_size = image_data.shape[0]

    patches = np.zeros(shape=(patch_size * patch_size, num_patches))

    for i in range(num_patches):
        image_id = random.randint(0, num_images - 1)
        image_x = random.randint(0, image_size - patch_size)
        image_y = random.randint(0, image_size - patch_size)

        img = image_data[:, :, image_id]
        patch = img[image_x:image_x + patch_size, image_y:image_y + patch_size].reshape(patch_size * patch_size)
        patches[:, i] = patch

    return patches
    
if __name__ == '__main__':
    patches = loadIMAGES_RAW()
    print patches


