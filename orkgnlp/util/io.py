"""
Includes Input/Output (I/O) functionalities like reading and writing from and into specific file formats.
"""

import json


def read_json(input_path):
    """
    Reads the json file of the given ``input_path``.

    :param input_path: Path to the json file
    :type input_path: str
    :return: A loaded json object.
    """
    with open(input_path, encoding='utf-8') as f:
        json_data = json.load(f)

    return json_data
