import asyncio
import glob
import os
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, MetaData

from config import config
from src.image_preprocessing import preprocess_image


# def setup_table(name, meta):
#     if not engine.dialect.has_table(sqlite_connection, name):
#         Table(
#             name,
#             meta,
#             Column("index", Integer, primary_key=True, autoincrement=True),
#             Column("image_title", String),
#             Column("vlad", String),
#             Column("comparison_2", String),
#             Column("comparison_3", String)
#         )


async def parse_image(image_name: str, table_name: str, engine) -> None:
    """async parse_image
    Parses a single image asynchronously to be entered into the database
    Calculates and saves the distinguishing factors of the image

    Parameters
    ----------
    image_name
    table_name
    engine

    Returns
    -------
    None
    """

    f_list = preprocess_image(image_name)

    if len(f_list) != 5:
        return

    v = f_list[0]
    h = f_list[1]
    b = f_list[2]
    w = f_list[3]
    f = f_list[4]

    columns = ["image_title", "vlad", "histogram", "feature_bagofwords", "haar_wavelet"]

    outer = []
    for i in w:
        inner = []
        for j in i:
            print(len(j))
            inner_inner = []
            for k in j:
                inner_inner.append(k)
            inner.append(inner_inner)
        outer.append(inner)

    img_data = pd.DataFrame(
        [
            [
                os.path.basename(image_name),
                str(v.tolist()),
                str(h.tolist()),
                str(b.tolist()),
                str(outer)
            ]
        ],
        columns=columns,
    )

    img_data.to_sql(table_name, engine, if_exists="append", index=False)


def initiate_database_creation() -> None:
    """initiate_database_creation
    Initiates creation of image_features database table.

    Returns
    -------
    None
    """
    images = glob.glob(os.path.join("..", config.config["datasets"]["images"], "*"))

    # Open a SQLite connection to put results into
    db_path = f'sqlite:///{Path(Path(__file__).parent.parent, config.config["database"]["path"])}'
    engine = create_engine(db_path, echo=True)

    meta = MetaData()
    # setup_table('image_features', meta)
    meta.create_all(engine)

    for image_name in images:
        asyncio.run(parse_image(image_name, "image_features", engine))


if __name__ == "__main__":
    initiate_database_creation()
