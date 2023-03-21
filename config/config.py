import os

import yaml

config = None
script_dir = os.path.dirname(__file__)


def get_path(value):
    return os.path.abspath(os.path.join(script_dir, "..", value))


filename = os.getenv("env", "config.yml").lower()
abs_file_path = os.path.join(script_dir, filename)
with open(abs_file_path, "r") as f:
    try:
        config = yaml.safe_load(f)
    except Exception as exc:
        print(exc)
