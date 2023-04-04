import json

import numpy as np
from sqlalchemy import text

from codebook import generate_codebook
from feature_extraction import get_features
from src.histogram_generation import generate_histogram
from vlad import create_vlad_vector


def similarity_score(query_img, test_name, connection):
    row = connection.execute(
        text("SELECT * FROM image_features WHERE image_title = :title"),
        {"title": test_name},
    ).fetchone()
    test_vlad_vector = np.array(json.loads(row[1]))
    test_histogram = np.array(json.loads(row[2]))

    features = get_features(query_img)
    query_codebook = generate_codebook(features["descriptors"])
    query_vlad_vector = create_vlad_vector(query_codebook, features["descriptors"])
    query_histogram = generate_histogram(query_img)

    vlad_distance = np.linalg.norm(query_vlad_vector - test_vlad_vector)
    vlad_score = 1 / (np.exp(vlad_distance))

    histogram_distance = np.linalg.norm(query_histogram - test_histogram)
    histogram_score = 1 / (np.exp(histogram_distance))

    print(vlad_score, histogram_score)
    score = vlad_score + histogram_score
    return score
