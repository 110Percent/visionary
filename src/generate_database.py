import asyncio
import glob
import os
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine, MetaData

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

async def parse_image(image_name: str, table_name: str, connection) -> None:
    """ async parse_image
    Parses a single image asynchronously to be entered into the database
    Calculates and saves the distinguishing factors of the image

    Parameters
    ----------
    image_name
    table_name
    connection

    Returns
    -------
    None
    """

    v, f = preprocess_image(image_name)
    columns = ['image_title', 'vlad', 'comparison_2', 'comparison_3']
    img_data = pd.DataFrame(
        [[os.path.basename(image_name), v, "", ""]],
        columns=columns
    )
    img_data.to_sql(
        table_name, connection, if_exists="append", index=False
    )


def initiate_database_creation() -> None:
    """ initiate_database_creation
    Initiates creation of image_features database table.

    Returns
    -------
    None
    """
    images = glob.glob("../data/images/*.jpg")

    # Open a SQLite connection to put results into
    db_path = f'sqlite:///{Path(Path(__file__).parent.parent, "images.sqlite")}'
    engine = create_engine(db_path, echo=True)
    sqlite_connection = engine.connect()

    meta = MetaData()
    # setup_table('image_features', meta)
    meta.create_all(engine)

    for image_name in images:
        asyncio.run(parse_image(image_name, "image_features", sqlite_connection))


if __name__ == '__main__':
    initiate_database_creation()
