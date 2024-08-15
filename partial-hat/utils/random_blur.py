import cv2
import numpy as np
import random

def apply_random_blur(image, blur_size, intensity):
    height, width, channel = image.shape
    start_x = random.randint(0, width - blur_size)
    start_y = random.randint(0, height - blur_size)
    end_x = start_x + blur_size
    end_y = start_y + blur_size
    original = image[start_y:end_y, start_x:end_x].copy()
    blur = cv2.GaussianBlur(original, (intensity, intensity), 0)
    ret = image.copy()
    ret[start_y:end_y, start_x:end_x] = blur
    mask = np.zeros((height, width))
    mask[start_y:end_y, start_x:end_x] = 255
    return (ret, mask)