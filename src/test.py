import os
import random
from pathlib import Path

import cv2
from sqlalchemy import create_engine, MetaData, text

from config import config
from src.comparison import similarity_score

img_dir = config.get_path(config.config["datasets"]["images"])


def get_random_image():
    filenames = os.listdir(img_dir)
    img_path = os.path.join(img_dir, random.choice(filenames))
    return img_path


def random_test():
    img_path = get_random_image()
    print(f"Running test on {img_path}")
    img = cv2.imread(img_path)

    """
    b, g, r = cv2.split(img)
    channels = [b, g, r]
    random.shuffle(channels)
    b, g, r = channels
    img = cv2.merge((b, g, r))
    cv2.imshow("Distorted", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """

    db_path = f'sqlite:///{Path(Path(__file__).parent.parent, config.config["database"]["path"])}'
    engine = create_engine(db_path)
    meta = MetaData()
    meta.create_all(engine)
    connection = engine.connect()

    max_score = 0
    closest_file = ""

    rows = connection.execute(text("SELECT * FROM image_features")).fetchall()
    for row in rows:
        name = row[0]
        score = similarity_score(img, name, connection)
        if score > max_score:
            max_score = score
            closest_file = name

    print(img_path)
    print(closest_file)
    print(img_path.split("/")[-1])
    success = img_path.endswith(closest_file)
    print(f"Success: {success}")
    print(f"Max Score: {max_score}")

    cv2.imshow("Real", cv2.imread(img_path))
    cv2.imshow("Guessed", cv2.imread(os.path.join(img_dir, closest_file)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


random_test()
