# -*- coding: utf-8 -*-
""" ResearchFieldClassifier encoder. """
import re
from typing import Any, Dict, Tuple

from overrides import overrides
from transformers import AutoTokenizer

from orkgnlp.common.service.base import ORKGNLPBaseEncoder


class ResearchFieldClassifierEncoder(ORKGNLPBaseEncoder):
    """
    The ResearchFieldClassifierEncoder encodes the given input
    to the arguments needed to execute the classification model.
    """

    def __init__(self):
        super().__init__()

        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained("malteos/scincl")
        self.max_input_sizes: int = 512
        self.device: str = "cpu"

    @overrides
    def encode(self, raw_input: Any, **kwargs: Any) -> Tuple[Any, Dict[str, Any]]:
        raw_input = " ".join(re.sub("<.*?>", " ", raw_input).split()).lower()
        input_encoding = self.tokenizer.encode(
            raw_input,
            padding="max_length",
            truncation=True,
            max_length=self.max_input_sizes,
            return_tensors="pt",
        )
        return [input_encoding], kwargs
