import numpy as np
from sklearn.cluster import KMeans


def create_vlad_vector(codebook, descriptors):
    kmeans = KMeans(n_clusters=codebook.shape[0], n_init=10)
    kmeans.fit(codebook)

    labels = kmeans.predict(descriptors)

    vlad_matrix = np.zeros_like(codebook)

    for i, center in enumerate(codebook):
        residuals = descriptors[labels == i] - center
        vlad_matrix[i] = residuals.sum(axis=0)

    vlad = vlad_matrix.flatten()
    vlad /= np.sqrt(np.sum(vlad**2))

    return vlad
