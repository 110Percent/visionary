import os

import cv2
import numpy as np

from config import config
from codebook import load_codebook, generate_codebook
from feature_extraction import get_features
from vlad import create_vlad_vector


def similarity_score(query_img, test_name):
    test_codebook = load_codebook(test_name)
    test_img_path = os.path.join(
        config.get_path(config.config["datasets"]["images"]), test_name
    )
    test_img = cv2.imread(test_img_path)
    test_features = get_features(test_img)
    test_vlad_vector = create_vlad_vector(test_codebook, test_features["descriptors"])

    features = get_features(query_img)
    query_codebook = generate_codebook(features["descriptors"])
    query_vlad_vector = create_vlad_vector(query_codebook, features["descriptors"])

    print("QUERY")
    print(query_vlad_vector)
    print(f"TEST {test_name}")
    print(test_vlad_vector)
    distance = np.linalg.norm(query_vlad_vector - test_vlad_vector)
    score = 1 / distance
    return score
