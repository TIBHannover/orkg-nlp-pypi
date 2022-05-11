"""
Includes Input/Output (I/O) functionalities like reading and writing from and into specific file formats.
"""

import json
import onnx

import pandas as pd


def read_json(input_path):
    """
    Reads the ``json`` file of the given ``input_path``.

    :param input_path: Path to the json file
    :type input_path: str
    :return: A loaded json object.
    """
    with open(input_path, encoding='utf-8') as f:
        json_data = json.load(f)

    return json_data


def read_df_from_json(input_path, key=None):
    """
    Reads the ``json`` file of the given ``input_path`` and converts it to pandas ``dataframe``.
    :param input_path: Path to the json file
    :type input_path: str
    :param key: Specifies the object to be converted.
    :type key: str
    :return: A loaded pandas dataframe object.
    """
    json_file = read_json(input_path)

    if key:
        return pd.json_normalize(json_file[key])

    return pd.json_normalize(json_file)


def read_onnx(input_path):
    """
    Reads the ``onnx`` file of the given ``input_path``

    :param input_path: Path to the onnx file
    :type input_path: str
    :return: A loaded onnx object.
    """
    return onnx.load(input_path)
