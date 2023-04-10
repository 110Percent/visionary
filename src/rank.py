from config import config
import cv2
from sqlalchemy import create_engine, MetaData, text
from pathlib import Path

from src.comparison import similarity_score
from src.image_preprocessing import preprocess_raw


def find_matches(img, n=5):
    db_path = f'sqlite:///{Path(Path(__file__).parent.parent, config.config["database"]["path"])}'
    engine = create_engine(db_path)
    meta = MetaData()
    meta.create_all(engine)
    connection = engine.connect()

    query_features = preprocess_raw(img)
    if len(query_features) < 4:
        print(query_features)

    rows = connection.execute(text("SELECT * FROM image_features")).fetchall()

    matches = []

    for row in rows:
        name = row[0]
        score = similarity_score(query_features, name, connection)
        entry = {"name": name, "score": score}
        if len(matches) < n:
            matches.append(entry)
        elif score > matches[-1]["score"]:
            matches[-1] = entry
        else:
            continue
        matches = sorted(matches, key=lambda e: e["score"] * -1)

    return matches
