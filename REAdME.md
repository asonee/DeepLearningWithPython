#Deep Learning With Python

This repository is a Python implementation version of **UFLDL**(**Unsupervised Feature Learning and Deep Learning**) tutorial exercises, the codes are passed the test and get the same results as exepected.

tutorial homepage: http://ufldl.stanford.edu/wiki/index.php/UFLDL_Tutorial

**IMPORTANT NOTES:** 

1. The datasets used in this repository can be found in UFLDL homepage.

2. I implememented softmax with bias, while the tutorial did not. I try softmax with bias and no bias, the result shows softmax with bias can achieved an exepected accuracy of 92.6% on MNIST dataset.  

3. Scipy 0.17 will lead to python kenerl died when load the *.mat format file. It is a inherent bug in scipy 0.17, so I do not recommend you to use scipy 0.17 to run the codes.

##Prerequisites
* python 2.7
* numpy
* scipy
* or just Anaconda (strongly recommend)

##Exercises' core source code file are listed as follows:

### Sparse Autoencoder
* sparseAutoencoder.py 
* sparseAutoencoderTest.py

### Preprocessing: PCA and Whitening
* whitening.py

### Softmax Regression
* softmax.py

### Self-Taught Learning and Unsupervised Feature Learning 
* self-taughtLearningTest.py

### Building Deep Networks for Classification(Stacked Sparse Autoencoder) 
* stackedAutoencoder.py
* stackedAutoencoderTest.py

### Linear Decoders with Autoencoders
* sparseAutoencoder.py
* SAEWithLinearDecoderTest.py

### Working with Large Images(Convolutional Neural Networks)
* cnn.py
* cnnTest.py

--- 
If you have any questions, please feel free to contact with me. (guanghuitu@gmail.com or guanghuitu@foxmail.com)

Enjoy it!


