import os
import random
from pathlib import Path

import cv2
import numpy as np
from sqlalchemy import create_engine, MetaData, text

from config import config
from src.comparison import similarity_score
from src.image_preprocessing import preprocess_raw
from src.rank import find_matches

img_dir = config.get_path(config.config["datasets"]["images"])


def get_random_image():
    db_path = f'sqlite:///{Path(Path(__file__).parent.parent, config.config["database"]["path"])}'
    engine = create_engine(db_path)
    meta = MetaData()
    meta.create_all(engine)
    connection = engine.connect()
    rows = connection.execute(
        text("SELECT image_title FROM image_features ORDER BY RANDOM() LIMIT 1")
    ).fetchall()
    img_path = os.path.join(img_dir, rows[0][0])
    return img_path


def random_test(**kwargs):
    default_kwargs = {
        "verbose": False,
        "shuffle_colourspace": False,
        "crop": False,
        "stretch": False,
        "flip": False,
    }
    kwargs = {**default_kwargs, **kwargs}

    img_path = get_random_image()

    if kwargs.get("verbose"):
        print(f"Running test on {img_path}")

    img = cv2.imread(img_path)

    if kwargs.get("shuffle_colourspace"):
        # Shuffle BGR channels
        b, g, r = cv2.split(img)
        channels = [b, g, r]
        random.shuffle(channels)
        b, g, r = channels
        img = cv2.merge((b, g, r))

    if kwargs.get("crop"):
        # Random crop
        crop = [
            random.randint(0, int(img.shape[0] / 4)),
            random.randint(0, int(img.shape[0] / 4)),
            random.randint(0, int(img.shape[1] / 4)),
            random.randint(0, int(img.shape[1] / 4)),
        ]
        img = img[crop[0] : img.shape[0] - crop[1], crop[2] : img.shape[1] - crop[3]]

    if kwargs.get("stretch"):
        # Random stretch
        w = random.randint(100, 800)
        h = random.randint(100, 800)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)

    if kwargs.get("flip"):
        # Random flip
        if random.randint(0, 1) == 1:
            img = cv2.flip(img, 1)
        if random.randint(0, 1) == 1:
            img = cv2.flip(img, -1)

    matches = find_matches(img)
    closest_file = matches[0]["name"]
    max_score = matches[0]["score"]

    success = False
    success_score = 0
    for i in range(len(matches)):
        if kwargs.get("verbose"):
            check = "✔️" if img_path.endswith(matches[i]["name"]) else ""
            print(f'{i + 1}. {matches[i]["name"]} - {matches[i]["score"]:.5f} {check}')
        if img_path.endswith(matches[i]["name"]):
            success = True
            success_score = (len(matches) - i) / len(matches)

    if kwargs.get("verbose"):
        cv2.imshow("Query", img)
        closest_img = cv2.imread(os.path.join(img_dir, closest_file))

        if success:
            cv2.rectangle(
                closest_img,
                (crop[2], crop[0]),
                (closest_img.shape[1] - crop[3], closest_img.shape[0] - crop[1]),
                (0, 255, 0),
                2,
            )

        cv2.imshow("Guessed", closest_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return success_score


def run_tests(n=50):
    results = []
    print(f"Running {n} uncorrupted tests")
    pure_tests = [random_test(stretch=True) for i in range(n)]
    results = [*results, *pure_tests]
    print_metrics(pure_tests)

    print(f"Running {n} shuffled colour space tests")
    colour_tests = [random_test(shuffle_colourspace=True) for i in range(n)]
    results = [*results, *colour_tests]
    print_metrics(colour_tests)

    print(f"Running {n} cropped tests")
    crop_tests = [random_test(crop=True) for i in range(n)]
    results = [*results, *crop_tests]
    print_metrics(crop_tests)

    print(f"Running {n} stretched tests")
    stretch_tests = [random_test(stretch=True) for i in range(n)]
    results = [*results, *stretch_tests]
    print_metrics(stretch_tests)

    print(f"Running {n} flipped tests")
    flip_tests = [random_test(flip=True) for i in range(n)]
    results = [*results, *flip_tests]
    print_metrics(flip_tests)

    print(f"Running {n * 2} randomized corruption tests")
    max_tests = [
        random_test(shuffle_colourspace=True, crop=True, stretch=True, flip=True)
        for i in range(n * 2)
    ]
    results = [*results, *max_tests]
    print_metrics(max_tests)

    print("\n=== FINAL RESULTS ===")
    print_metrics(results)


def print_metrics(results):
    found_score = (sum(result > 0 for result in results)) / len(results)
    print(f"Percentage where Top 5: {(found_score * 100):.2f}%")

    perfect_score = (sum(result == 1 for result in results)) / len(results)
    print(f"Percentage where #1: {(perfect_score * 100):.2f}")

    print(f"Average score: {(np.mean(results) * 100):.2f}%")
    print("")


run_tests()
