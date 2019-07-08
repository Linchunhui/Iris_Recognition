# -*- coding: utf-8 -*-


import numpy as np
import cv2
from skimage.transform import hough_circle, hough_circle_peaks


def IrisLocalization(eye):
    blured = cv2.bilateralFilter(eye, 9, 100, 100)
    Xp = blured.sum(axis=0).argmin()
    Yp = blured.sum(axis=1).argmin()
    x = blured[max(Yp - 60, 0):min(Yp + 60, 280), max(Xp - 60, 0):min(Xp + 60, 320)].sum(axis=0).argmin()
    y = blured[max(Yp - 60, 0):min(Yp + 60, 280), max(Xp - 60, 0):min(Xp + 60, 320)].sum(axis=1).argmin()
    Xp = max(Xp - 60, 0) + x
    Yp = max(Yp - 60, 0) + y
    if Xp >= 100 and Yp >= 80:
        blur = cv2.GaussianBlur(eye[Yp - 60:Yp + 60, Xp - 60:Xp + 60], (5, 5), 0)
        pupil_circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=200, param1=200, param2=12,
                                         minRadius=15, maxRadius=80)
        xp, yp, rp = np.round(pupil_circles[0][0]).astype("int")
        xp = Xp - 60 + xp
        yp = Yp - 60 + yp
    else:
        pupil_circles = cv2.HoughCircles(blured, cv2.HOUGH_GRADIENT, 4, 280, minRadius=25, maxRadius=55, param2=51)
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
    return np.array(iris), np.array([xp, yp, rp])


if __name__=="__main__":
    eye = r"D:/study/iris/CASIA-Iris-Thousand\779\R\S5779R00.jpg"
    img = cv2.imread(eye, 0)
    # a,b=IrisLocalization(img)
    a1, b1 = IrisLocalization(img)
