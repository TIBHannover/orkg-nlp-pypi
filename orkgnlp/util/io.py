import json


def read_json(input_path):
    with open(input_path, encoding='utf-8') as f:
        json_data = json.load(f)

    return json_data
