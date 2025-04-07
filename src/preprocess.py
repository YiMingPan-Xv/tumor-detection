import numpy as np
from PIL import Image

from skimage.util import random_noise
from skimage.transform import rotate


def resize_and_gray(images, size=(128, 128)):
    resized_images = []
    for image in images:
        gray_image = Image.fromarray(image).convert("L")
        resized_image = np.array(gray_image.resize(size))
        resized_images.append(resized_image)
    return resized_images

def augment(images, labels, num_augmented=2):
    augmented_images = []
    if images.shape[0] != labels.shape[0]:
        raise ValueError(f"Mismatch between length of images and labels!\n{images.shape[0]} != {labels.shape[0]}")
    for image, label in zip(images, labels):
        for _ in range(num_augmented):
            # Apply random rotation
            rotated = rotate(image, angle=np.random.uniform(-30, 30), mode='wrap')
            # Add random noise
            noisy = random_noise(rotated, mode='gaussian', var=0.01)
            # Convert back to uint8
            augmented_images.append(((noisy * 255).astype(np.uint8), label))
    return augmented_images