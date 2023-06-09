import json

import numpy as np
from numba import jit
from sqlalchemy import text


@jit
def compare(
    query_vlad_vector,
    query_histogram,
    query_bagowords,
    test_vlad_vector,
    test_histogram,
    test_bagowords,
):
    vlad_distance = np.linalg.norm(query_vlad_vector - test_vlad_vector)
    vlad_score = 0.5 / (np.exp(vlad_distance))

    histogram_distance = np.linalg.norm(query_histogram - test_histogram)
    histogram_score = 1 / (np.exp(histogram_distance))

    bagowords_distance = np.linalg.norm(query_bagowords - test_bagowords)
    bagowords_score = 0.3 / (np.exp(bagowords_distance))

    score = vlad_score + histogram_score + bagowords_score
    return score


def similarity_score(query_data, test_name, connection):
    row = connection.execute(
        text("SELECT * FROM image_features WHERE image_title = :title"),
        {"title": test_name},
    ).fetchone()
    test_vlad_vector = np.array(json.loads(row[1]))
    test_histogram = np.array(json.loads(row[2]))
    test_bagowords = np.array(json.loads(row[3]))
    [query_vlad_vector, query_histogram, query_bagowords, features] = query_data
    # print(test_name, vlad_score, histogram_score, score)
    return compare(
        query_vlad_vector,
        query_histogram,
        query_bagowords,
        test_vlad_vector,
        test_histogram,
        test_bagowords,
    )
