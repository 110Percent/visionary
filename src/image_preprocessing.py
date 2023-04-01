import os

import cv2

from config import config
from src import feature_extraction
from codebook import generate_codebook
from vlad import create_vlad_vector

'''
def difference_score(features1, feature2):
    """
    Difference score between two features
    """
    dist_diff = abs(features1["relative_dist"] - feature2["relative_dist"])
    max_response_diff = abs(features1["max_response"] - feature2["max_response"])
    max_response_distance = math.dist(
        features1["relative_max_response"], feature2["relative_max_response"]
    )
    mean_distance = math.dist(features1["mean_feature"], feature2["mean_feature"])

    weights = [1, 1, 2, 1]
    score = np.average(
        [dist_diff, max_response_diff, max_response_distance, mean_distance],
        weights=weights,
    )
    return score

'''


def preprocess_images():
    """
    Preprocess images
    """
    print("Preprocessing images...")
    feature_list = []
    image_dir = config.get_path(config.config["datasets"]["images"])
    img_total = len(os.listdir(config.get_path(config.config["datasets"]["images"])))
    images_processed = 0
    for root, dirs, files in os.walk(image_dir):
        for name in files:
            if ".npy" in name:
                continue
            filename = os.path.join(root, name)
            print(name)
            feature_list.append(preprocess_image(filename)[1])
            images_processed += 1
            if images_processed % 20 == 0:
                print(f"Processed {images_processed}/{img_total} images")
            """
            diff = difference_score(bear_features, features)
            if diff < match_distance:
                match = filename
                match_distance = diff
            """

    """
    match_features = feature_extraction.get_features(cv2.imread(match))
    print("MATCH", match)
    print(match_features)
    bear_img = cv2.imread(bear)
    height = min(500, int(bear_img.shape[0] / bear_img.shape[1] * 500))
    bear_img = cv2.resize(
        bear_img, (int(bear_img.shape[1] / bear_img.shape[0] * height), height)
    )
    match_img = cv2.imread(match)
    match_img = cv2.resize(
        match_img, (int(match_img.shape[1] / match_img.shape[0] * height), height)
    )
    cv2.imshow("match", cv2.hconcat([bear_img, match_img]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
    return feature_list


def preprocess_image(name: str):
    """
    Preprocess image
    """
    image = cv2.imread(name)
    features = feature_extraction.get_features(image)

    codebook = generate_codebook(features["descriptors"], os.path.basename(name), write=True)
    v = create_vlad_vector(codebook, features["descriptors"])

    return v, features


if __name__ == '__main__':
    preprocess_images()
