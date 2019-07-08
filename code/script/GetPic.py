#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
from Segmentation import IrisLocalization
from Normalization import IrisNormalization
from Enhancement import ImageEnhancement
from Gabor import FeatureExtraction
import datetime
import pandas as pd

train = pd.read_csv("D:/study/iris/csv/train1.csv")
test = pd.read_csv("D:/study/iris/csv/test1.csv")

train_img_list = train["img"]
train_label_list = train["label"]
test_img_list = test["img"]
test_label_list = test["label"]
train_size = len(train_img_list)
test_size = len(test_img_list)

train_features = np.zeros((train_size,1536))
train_classes = np.zeros(train_size, dtype = np.uint8)
test_features = np.zeros((test_size,1536))
test_classes = np.zeros(test_size, dtype = np.uint8)

starttime = datetime.datetime.now()

'''for i in range(train_size):
    train_path=train_img_list[i]
    #print(train_path)
    img_name=train_path.split('\\')[-1].split('.')[0]
    img = cv2.imread(train_path, 0)
    iris, pupil = IrisLocalization(img)
    #cv2.circle(img, (iris[0], iris[1]), iris[2], (0, 0, 255), 1)
    #cv2.circle(img, (pupil[0], pupil[1]), pupil[2], (0, 255, 0), 1)
    #cv2.imwrite("D:/study/iris/process_data/circle/train/{}.jpg".format(img_name),img)
    normalized = IrisNormalization(img, pupil, iris)
    cv2.imwrite("D:/study/iris/process_data/rectangle/train/{}.jpg".format(img_name),normalized)
    ROI = ImageEnhancement(normalized)
    cv2.imwrite("D:/study/iris/process_data/enhancement/train/{}.jpg".format(img_name), ROI)
    #train_features[i, :] = FeatureExtraction(ROI)
    #train_classes[i] = train_label_list[i]
    print('train_feature:',train_img_list[i],train_label_list[i],'{}/{}'.format(i,train_size))'''

for j in range(test_size):
    test_path=test_img_list[j]
    img_name = test_path.split('\\')[-1].split('.')[0]
    img = cv2.imread(test_path, 0)
    iris, pupil = IrisLocalization(img)
    #cv2.circle(img, (iris[0], iris[1]), iris[2], (0, 0, 255), 1)
    #cv2.circle(img, (pupil[0], pupil[1]), pupil[2], (0, 255, 0), 1)
    #cv2.imwrite("D:/study/iris/process_data/circle/test/{}".format(img_name), img)
    normalized = IrisNormalization(img, pupil, iris)
    cv2.imwrite("D:/study/iris/process_data/rectangle/test/{}.jpg".format(img_name), normalized)
    #ROI = ImageEnhancement(normalized)
    #cv2.imwrite("D:/study/iris/process_data/enhancement/test/{}.jpg".format(img_name), ROI)
    #test_features[j, :] = FeatureExtraction(ROI)
    #test_classes[j] = test_label_list[j]
    print('test_feature:', test_img_list[j], test_label_list[j], '{}/{}'.format(j, test_size))

endtime = datetime.datetime.now()