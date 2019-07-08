# Iris_Recognition
Iris recognition include tradition algorithm and deep learning.

# Dataset
## CASIA-Iris version1.0
Include 108 classes, each class has 7 images, three of them for train and the other for test.
 
## CASIA-Iris-Thousand
Include 1000 classes, each class has 20 images,half are left eyes and half right.
We random select 70% of them for train, and 30% for test.All the results are based on this dataset.

# Algorithm
## 1.tradition algorithm
### Preprocessing
We use hough transform and canny edge detection to segment the iris,and then unfold the ring between the outer circle and inner circle into a rectangle of size 64*512. After normalization,we did local image equalization as Li Ma's paper.Finally,we use gabor filter to extract the feature vector from the ROI.
 
#### Segmentation.py
#### Normalization.py
#### Enhancement.py

### USIT v2.2
We also recommend an open-source software,USIT v2.2,from the University of Salzburg to complete the preprocessing.
[Github](https://github.com/ngoclamvt123/usit-v2.2.0)
You just need to clone the git and install opencv and boost,and then release wahet.cpp. 

usage:  
for a single image  
test.exe -i D:\study\iris\CASIA-Iris-Thousand\000\L\S5000L00.jpg -o texture.png -s 256 64 -e

for a batch images  
test.exe -i D:\study\iris\CASIA\origin\train\*.jpg  -o D:/study/iris/CASIA/enhance_512/train/?1.jpg  -s 512 64 -e

If you don't need enhancement,you just need delete "-e". 
If you need the segmentation,you just need add "-sr D:/study/iris/CASIA/seg/train/?1.jpg"
 
### Gabor Feature Extraction
We use gabor filter to complete the feature extraction.
 
### Distance Based Match 
We use cityblock distance,euclidean distance and cosine distance to match,and results respectively are 88.19%,84.95% and 85.42%.

### Machine Learning Predict
We use pca to reduce the dimension and then use KNN and SVM to train and predict.When the dimension reduce to 380, the result is the best,90.2% for KNN and 90.7 for SVM.

## 2.CNN Feature Extraction
We use InceptionV4,ResNet-101,Densenet121 to extract feature from the ROI after enhancement.When inceptionV4 at "Mixed6a",ResNet with block[3,4,9] and DenseNet with block[6,12,3] and then use pca reduce the dimension to 580 for SVM to get the best results 95.8%,96.4% and 97.1%.We also append avgrage pooling after convolution to avoid MemoryError.

## 3.CNN Classification
We also use a mini densenet with 40 layers to train a model.Dataset are the ROI of the CASIA-Iris-Thousand.However,it doesn't work for the limitation of the dataset and lead to over-fit.





