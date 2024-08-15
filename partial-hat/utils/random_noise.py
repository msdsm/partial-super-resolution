import cv2
import numpy as np
import random

def apply_random_noise(image, noise_size, sigma):
    height, width, channel = image.shape
    start_x = random.randint(0, width - noise_size)
    start_y = random.randint(0, height - noise_size)
    end_x = start_x + noise_size
    end_y = start_y + noise_size
    original = image[start_y:end_y, start_x:end_x].copy()
    noise = np.random.normal(0, sigma, original.shape)
    noisy_img = original.astype(np.float64) + noise
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    ret = image.copy()
    ret[start_y:end_y, start_x:end_x] = noisy_img
    mask = np.zeros((height, width))
    mask[start_y:end_y, start_x:end_x] = 255
    return (ret, mask)