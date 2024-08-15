import os
import cv2
import numpy as np
import random
import os
from PIL import Image
import numpy as np
import random

def apply_random_pixelization(image, pixelization_size, intensity):
    height, width, channel = image.shape
    start_x = random.randint(0, width - pixelization_size)
    start_y = random.randint(0, height - pixelization_size)
    end_x = start_x + pixelization_size
    end_y = start_y + pixelization_size
    original = image[start_y:end_y, start_x:end_x].copy()
    small = cv2.resize(
        original,
        (
            round(pixelization_size / intensity),
            round(pixelization_size / intensity)
        )
    )
    pixelization = cv2.resize(
        small,
        (
            pixelization_size,
            pixelization_size
        ),
        interpolation=cv2.INTER_NEAREST
    )
    ret = image.copy()
    ret[start_y:end_y, start_x:end_x] = pixelization
    mask = np.zeros((height, width))
    mask[start_y:end_y, start_x:end_x] = 255
    return (ret, mask)