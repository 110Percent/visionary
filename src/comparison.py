import json

import numpy as np
from sqlalchemy import text

from codebook import generate_codebook
from feature_extraction import get_features
from vlad import create_vlad_vector


def similarity_score(query_img, test_name, connection):
    row = connection.execute(
        text("SELECT * FROM image_features WHERE image_title = :title"),
        {"title": test_name},
    ).fetchone()
    test_vlad_vector = np.array(json.loads(row[1]))

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
