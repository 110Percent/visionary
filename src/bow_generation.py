import cv2
import numpy as np


def generate_bow(img: np.ndarray) -> np.ndarray:

    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)

    k = 20  # number of visual words
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.01)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, labels, centroids = cv2.kmeans(descriptors, k, None, criteria, 10, flags)

    bow_vector = np.zeros(k)
    for label in labels:
        bow_vector[label] += 1

    bow_vector = bow_vector / np.linalg.norm(bow_vector)

    return bow_vector
