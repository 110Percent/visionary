import cv2
import numpy as np
import pywt


def create_haar_wavelet_vector(img):
    """

    Parameters
    ----------
    img

    Returns
    -------

    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rows, cols = gray.shape

    gray = gray[:rows // 2 * 2, :cols // 2 * 2]
    query_ll = gray[0:rows // 2, 0:cols // 2]
    query_lh = gray[0:rows // 2, cols // 2:cols]
    query_hl = gray[rows // 2:rows, 0:cols // 2]
    query_hh = gray[rows // 2:rows, cols // 2:cols]

    query_ll = cv2.filter2D(query_ll, -1, np.array([[0.5, 0.5], [0.5, 0.5]]))
    query_lh = cv2.filter2D(query_lh, -1, np.array([[-0.5, 0.5], [-0.5, 0.5]]))
    query_hl = cv2.filter2D(query_hl, -1, np.array([[-0.5, -0.5], [0.5, 0.5]]))
    query_hh = cv2.filter2D(query_hh, -1, np.array([[-0.5, 0.5], [0.5, -0.5]]))

    return [query_ll, query_lh, query_hl, query_hh]
