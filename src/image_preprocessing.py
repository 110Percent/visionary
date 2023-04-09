import os

import cv2

from codebook import generate_codebook
from config import config
from src import feature_extraction
from src.bow_generation import generate_bow
from src.histogram_generation import generate_histogram
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
            preprocessed = preprocess_image(filename)[1]
            if preprocessed is None:
                continue
            feature_list.append(preprocessed)
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
    return preprocess_raw(image)


def preprocess_raw(image):
    features = feature_extraction.get_features(image)

    if features["descriptors"].shape[0] < config.config["codebook"]["clusters"]:
        return None, None

    codebook = generate_codebook(features["descriptors"])
    v = create_vlad_vector(codebook, features["descriptors"])

    histogram = generate_histogram(image)

    bow = generate_bow(image)

    return [v, histogram, bow, features]


if __name__ == "__main__":
    preprocess_images()
