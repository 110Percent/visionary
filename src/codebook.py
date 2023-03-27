import os.path

import cv2
import numpy as np

from config import config


def write_codebook(codebook, name):
    out_dir = config.get_path(os.path.join('data', 'codebooks'))
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, name + '.npy'), codebook)


def generate_codebook(descriptors, name):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.01)
    flags = cv2.KMEANS_PP_CENTERS
    _, _, codebook = cv2.kmeans(descriptors, config.config['codebook']['clusters'], None, criteria, 10,
                                               flags)
    write_codebook(codebook, name)
    return codebook
