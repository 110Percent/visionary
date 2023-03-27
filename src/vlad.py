import numpy as np


def get_closest_centroid(codebook, descriptor):
    lowest_distance = 999999
    lowest_index = -1
    for i in range(codebook.shape[0]):
        distance = np.linalg.norm(codebook[i] - descriptor)
        if distance < lowest_distance:
            lowest_distance = distance
            lowest_index = i
    return codebook[lowest_index]


def create_vlad_vector(codebook, descriptors):
    k = descriptors.shape[0]  # number of visual words
    d = descriptors.shape[1]  # dimension of SIFT descriptor
    vlad = np.zeros((k * d,), dtype=np.float32)

    for i in range(descriptors.shape[0]):
        word = descriptors[i]
        centroid = get_closest_centroid(codebook, word)
        residual = word - centroid
        vlad[i * d:(i + 1) * d] = residual

    return vlad
