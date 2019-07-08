import math
import numpy as np
import cv2
from PIL import Image
from skimage.filters.rank import equalize
from skimage.morphology import disk

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


