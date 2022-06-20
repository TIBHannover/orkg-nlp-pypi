"""
Includes Input/Output (I/O) functionalities like reading and writing from and into specific file formats.
"""

import json
import pickle
import torch
import onnx

import pandas as pd
from onnx import ModelProto
from pandas import DataFrame
from typing import Dict, Any, Type

from torch.jit import ScriptModule
from transformers import PreTrainedModel


def read_json(input_path: str) -> Dict[str, Any]:
    """
    Reads the ``json`` file of the given ``input_path``.

    :param input_path: Path to the json file
    :return: A loaded json object.
    """
    with open(input_path, encoding='utf-8') as f:
        json_data = json.load(f)

    return json_data


def read_df_from_json(input_path: str, key: str = None) -> DataFrame:
    """
    Reads the ``json`` file of the given ``input_path`` and converts it to pandas ``dataframe``.

    :param input_path: Path to the json file
    :param key: Specifies the object to be converted.
    :return: A loaded pandas dataframe object.
    """
    json_file = read_json(input_path)

    if key:
        return pd.json_normalize(json_file[key])

    return pd.json_normalize(json_file)


def read_onnx(input_path: str) -> ModelProto:
    """
    Reads the ``onnx`` file of the given ``input_path``.

    :param input_path: Path to the onnx file
    :return: A loaded onnx object.
    """
    return onnx.load(input_path)


def read_pickle(input_path: str) -> Any:
    """
    Reads the ``pickle`` file of the given ``input_path``.

    :param input_path: Path to the pickle file
    :return: A loaded pickle object.
    """
    with open(input_path, 'rb') as f:
        loaded_object = pickle.load(f)

    return loaded_object


def read_csv(input_path: str, header: str = None, sep: str = ',') -> DataFrame:
    """
    Reads the ``csv`` file of the given ``input_path``.

    :param input_path: Path to the csv file.
    :param header: See pandas.read_csv. Defaults to None
    :param sep: See pandas.read_csv. Defaults to ','
    :return: str
    """
    return pd.read_csv(input_path, header=header, sep=sep)


def load_torch_jit(input_path: str, device_name: str = 'cpu') -> ScriptModule:
    """
    Loads the scripted/traced ``torch model`` file (ScriptModule) of the given ``input_path``.

    :param input_path: Path to the scripted/traced torch model file (ScriptModule).
    :param device_name: Defaults to ``cpu``
    :return: A loaded torch model (ScriptModule) object.
    """

    return torch.jit.load(input_path, map_location=torch.device(device_name))


def load_transformers_pretrained(input_path: str, transformers_cls: Type[PreTrainedModel]) -> PreTrainedModel:
    """
    Loads a transformers model weights given of the given ``input_path`` to the given ``transformers_cls``.

    :param input_path: Path to the pretrained transformers model folder
    :param transformers_cls: The class type of the pretrained model you need to load.
    :return:
    """

    return transformers_cls.from_pretrained(input_path)
