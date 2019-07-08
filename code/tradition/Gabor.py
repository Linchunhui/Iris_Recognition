from scipy import ndimage
import math
import numpy as np


###this is feature extraction and works
# defined filter
def defined_filter(x, y, f):
    m1 = math.cos(2 * math.pi * f * math.sqrt(x ** 2 * y ** 2))
    return m1


space_constant_x1 = 3
space_constant_x2 = 4.5
space_constant_y = 1.5

f1 = 0.1
f2 = 0.07
x1 = range(-9, 10, 1)
x2 = range(-14, 15, 1)
y = range(-5, 6, 1)


def gabor_filter(x, y, space_constant_x, space_constant_y, f):
    m1 = defined_filter(x, y, f)
    return (1 / (2 * math.pi * space_constant_x * space_constant_y)
            * np.exp(-1 / 2 * (x ** 2 / (space_constant_x ** 2) + y ** 2 / (space_constant_y ** 2)))) * m1


def FeatureExtraction(roi):
    filter1 = []
    filter2 = []
    f1 = 0.1
    f2 = 0.07
    x1 = range(-9, 10, 1)
    x2 = range(-14, 15, 1)
    y = range(-5, 6, 1)
    space_constant_x1 = 3
    space_constant_x2 = 4.5
    space_constant_y = 1.5
    for j in range(len(y)):
        for i in range(len(x1)):
            cell_1 = gabor_filter(x1[i], y[j], space_constant_x1, space_constant_y, f1)
            filter1.append(cell_1)
        for k in range(len(x2)):
            cell_2 = gabor_filter(x2[k], y[j], space_constant_x2, space_constant_y, f2)
            filter2.append(cell_2)
    filter1 = np.reshape(filter1, (len(y), len(x1)))
    filter2 = np.reshape(filter2, (len(y), len(x2)))

    filtered_eye1 = ndimage.convolve(roi, np.real(filter1), mode='wrap', cval=0)
    filtered_eye2 = ndimage.convolve(roi, np.real(filter2), mode='wrap', cval=0)

    vector = []
    i = 0
    while i < roi.shape[0]:
        j = 0
        while j < roi.shape[1]:
            mean1 = filtered_eye1[i:i + 8, j:j + 8].mean()
            mean2 = filtered_eye2[i:i + 8, j:j + 8].mean()
            AAD1 = abs(filtered_eye1[i:i + 8, j:j + 8] - mean1).mean()
            AAD2 = abs(filtered_eye2[i:i + 8, j:j + 8] - mean2).mean()

            vector.append(mean1)
            vector.append(AAD1)
            vector.append(mean2)
            vector.append(AAD2)
            j = j + 8
        i = i + 8
    vector = np.array(vector)
    return vector

