import math

import cv2
import numpy as np

brightness = 0.5


def get_features(img):
    buffer = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=20)
    sift = cv2.SIFT_create()
    key_points = sift.detect(gray, None)
    max_response_point = max(key_points, key=lambda x: x.response)
    key_points = [
        point
        for point in key_points
        if point.response > (max_response_point.response * 0.5)
    ]

    weights = [point.size for point in key_points]
    mean_point = (
        int(np.average([point.pt[0] for point in key_points], weights=weights)),
        int(np.average([point.pt[1] for point in key_points], weights=weights)),
    )
    for point in key_points:
        cv2.circle(
            buffer,
            (int(point.pt[0]), int(point.pt[1])),
            int(point.response * point.size * 10),
            (0, 255, 0),
            2,
        )
    cv2.circle(buffer, mean_point, 10, (0, 0, 255), -1)
    cv2.circle(
        buffer,
        (int(max_response_point.pt[0]), int(max_response_point.pt[1])),
        10,
        (255, 0, 0),
        -1,
    )

    relative_mean_point = (mean_point[0] / img.shape[1], mean_point[1] / img.shape[0])

    relative_max_response_point = (
        max_response_point.pt[0] / img.shape[1],
        max_response_point.pt[1] / img.shape[0],
    )

    relative_dist = math.dist(relative_mean_point, relative_max_response_point)

    """
    cv2.imshow("img", buffer)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
    return {
        "relative_dist": relative_dist,
        "relative_max_response": relative_max_response_point,
        "max_response": max_response_point.response,
        "mean_feature": relative_mean_point,
    }
