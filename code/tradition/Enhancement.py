from skimage.filters.rank import equalize
from skimage.morphology import disk
import numpy as np


def ImageEnhancement(normalized_iris):
    row = 64
    col = 512
    normalized_iris = normalized_iris.astype(np.uint8)

    enhanced_image = normalized_iris

    enhanced_image = equalize(enhanced_image, disk(32))

    roi = enhanced_image[0:48, :]
    return roi
