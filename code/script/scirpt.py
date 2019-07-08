#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import math
import numpy as np
import cv2
from PIL import Image
from skimage.filters.rank import equalize
from skimage.morphology import disk
from skimage.transform import hough_circle, hough_circle_peaks
import pandas as pd

def IrisLocalization1(eye):
    blured = cv2.bilateralFilter(eye, 9, 100, 100)
    Xp = blured.sum(axis=0).argmin()
    Yp = blured.sum(axis=1).argmin()
    x = blured[max(Yp - 60, 0):min(Yp + 60, 480), max(Xp - 60, 0):min(Xp + 60, 640)].sum(axis=0).argmin()
    y = blured[max(Yp - 60, 0):min(Yp + 60, 480), max(Xp - 60, 0):min(Xp + 60, 640)].sum(axis=1).argmin()
    Xp = max(Xp - 60, 0) + x
    Yp = max(Yp - 60, 0) + y
    if Xp >= 200 and Yp >= 160:
        blur = cv2.GaussianBlur(eye[Yp - 60:Yp + 60, Xp - 60:Xp + 60], (5, 5), 0)
        pupil_circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=200, param1=200, param2=12,
                                         minRadius=15, maxRadius=80)
        xp, yp, rp = np.round(pupil_circles[0][0]).astype("int")
        xp = Xp - 60 + xp
        yp = Yp - 60 + yp
    else:
        pupil_circles = cv2.HoughCircles(blured, cv2.HOUGH_GRADIENT, 4, 480, minRadius=25, maxRadius=55, param2=51)
        xp, yp, rp = np.round(pupil_circles[0][0]).astype("int")
    eye_copy = eye.copy()
    rp = rp + 7  # slightly enlarge the pupil radius makes a better result
    blured_copy = cv2.medianBlur(eye_copy, 11)
    blured_copy = cv2.medianBlur(blured_copy, 11)
    blured_copy = cv2.medianBlur(blured_copy, 11)
    eye_edges = cv2.Canny(blured_copy, threshold1=15, threshold2=30, L2gradient=True)
    eye_edges[:, xp - rp - 30:xp + rp + 30] = 0

    hough_radii = np.arange(rp + 45, 150, 2)
    hough_res = hough_circle(eye_edges, hough_radii)
    accums, xi, yi, ri = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=1)
    iris = []
    iris.extend(xi)
    iris.extend(yi)
    iris.extend(ri)
    if ((iris[0] - xp) ** 2 + (iris[1] - yp) ** 2) ** 0.5 > rp * 0.3:
        iris[0] = xp
        iris[1] = yp
    #cv2.imshow('blur',blured)
    cv2.circle(blured,(iris[0],iris[1]),iris[2],(0,0,255),1)

    cv2.circle(blured, (xp, yp), rp, (0, 255,0 ), 1)
    #cv2.imshow('circle1',blured)
    cv2.imshow('circle2',blured)
    cv2.waitKey(0)
    return np.array(iris), np.array([xp, yp, rp])

def IrisLocalization(eye):
    blured = cv2.bilateralFilter(eye, 9, 100, 100)
    Xp = blured.sum(axis=0).argmin()
    Yp = blured.sum(axis=1).argmin()
    x = blured[max(Yp - 120, 0):min(Yp + 120, 480), max(Xp - 120, 0):min(Xp + 120, 640)].sum(axis=0).argmin()
    y = blured[max(Yp - 120, 0):min(Yp + 120, 480), max(Xp - 120, 0):min(Xp + 120, 640)].sum(axis=1).argmin()
    Xp = max(Xp - 120, 0) + x
    Yp = max(Yp - 120, 0) + y
    if Xp >= 200 and Yp >= 160:
        blur = cv2.GaussianBlur(eye[Yp - 120:Yp + 120, Xp - 120:Xp + 120], (5, 5), 0)
        pupil_circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=200, param1=210, param2=12,
                                         minRadius=10, maxRadius=80)
        xp, yp, rp = np.round(pupil_circles[0][0]).astype("int")
        xp = Xp - 120 + xp
        yp = Yp - 120 + yp
    else:
        pupil_circles = cv2.HoughCircles(blured, cv2.HOUGH_GRADIENT, 4, 200, minRadius=25, maxRadius=80, param2=25)
        xp, yp, rp = np.round(pupil_circles[0][0]).astype("int")
    eye_copy = eye.copy()
    rp = rp + 7  # slightly enlarge the pupil radius makes a better result
    blured_copy = cv2.medianBlur(eye_copy, 11)
    blured_copy = cv2.medianBlur(blured_copy, 11)
    blured_copy = cv2.medianBlur(blured_copy, 11)
    eye_edges = cv2.Canny(blured_copy, threshold1=15, threshold2=30, L2gradient=True)
    eye_edges[:, xp - rp - 60:xp + rp + 60] = 0

    hough_radii = np.arange(rp + 45, 300, 2)
    hough_res = hough_circle(eye_edges, hough_radii)
    accums, xi, yi, ri = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=1)
    iris = []
    iris.extend(xi)
    iris.extend(yi)
    iris.extend(ri)
    if ((iris[0] - xp) ** 2 + (iris[1] - yp) ** 2) ** 0.5 > rp * 0.3:
        iris[0] = xp
        iris[1] = yp
    #cv2.imshow('blur',blured)
    cv2.circle(blured,(iris[0],iris[1]),iris[2],(0,0,255),1)

    cv2.circle(blured, (xp, yp), rp, (0, 255,0 ), 1)
    #cv2.imshow('circle1',blured)
    cv2.imshow('circle2',blured)
    cv2.waitKey(0)
    return np.array(iris), np.array([xp, yp, rp])


def IrisNormalization(image, inner_circle, outer_circle):
    localized_img = image
    row = 64
    col = 512
    normalized_iris = np.zeros(shape=(64, 512))
    inner_y = inner_circle[0]  # height
    inner_x = inner_circle[1]  # width
    outer_y = outer_circle[0]
    outer_x = outer_circle[1]
    angle = 2.0 * math.pi / col
    inner_boundary_x = np.zeros(shape=(1, col))
    inner_boundary_y = np.zeros(shape=(1, col))
    outer_boundary_x = np.zeros(shape=(1, col))
    outer_boundary_y = np.zeros(shape=(1, col))
    for j in range(col):
        inner_boundary_x[0][j] = inner_circle[0] + inner_circle[2] * math.cos(angle * (j))
        inner_boundary_y[0][j] = inner_circle[1] + inner_circle[2] * math.sin(angle * (j))

        outer_boundary_x[0][j] = outer_circle[0] + outer_circle[2] * math.cos(angle * (j))
        outer_boundary_y[0][j] = outer_circle[1] + outer_circle[2] * math.sin(angle * (j))

    for j in range(512):
        for i in range(64):
            normalized_iris[i][j] = localized_img[min(int(int(inner_boundary_y[0][j])
                                                          + (int(outer_boundary_y[0][j]) - int(
                inner_boundary_y[0][j])) * (i / 64.0)), localized_img.shape[0] - 1)][min(int(int(inner_boundary_x[0][j])
                                                                                             + (int(
                outer_boundary_x[0][j]) - int(inner_boundary_x[0][j]))
                                                                                             * (i / 64.0)),
                                                                                         localized_img.shape[1] - 1)]

    res_image = 255 - normalized_iris
    return res_image

def MatrixToImage(data):
    data = data*255
    new_im = Image.fromarray(data.astype(np.uint8))
    return new_im

def ImageEnhancement(normalized_iris):
    row = 64
    col = 512
    normalized_iris = normalized_iris.astype(np.uint8)

    enhanced_image = normalized_iris

    enhanced_image = equalize(enhanced_image, disk(32))

    roi = enhanced_image[0:48, :]
    return roi

if __name__=="__main__":
    train = pd.read_csv("D:/study/iris/csv/train.csv")
    train_img_list = train["img"]
    for i in range(23,len(train_img_list)):
        eye = train_img_list[i]
        print(i)
        img = cv2.imread(eye, 0)
        # a,b=IrisLocalization(img)
        a1, b1 = IrisLocalization(img)