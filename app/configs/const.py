import os
import json


file_path = 'package.json'


if os.path.exists(file_path):
    with open(file_path, 'r') as file:
        FILEPATH = json.load(file)
else:
    raise FileNotFoundError
