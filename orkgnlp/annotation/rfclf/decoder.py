# -*- coding: utf-8 -*-
""" ResearchFieldClassifier decoder. """
from typing import Any, Generator, Union

import torch
from overrides import overrides

from orkgnlp.common.service.base import ORKGNLPBaseDecoder


class ResearchFieldClassifierDecoder(ORKGNLPBaseDecoder):
    """
    The ResearchFieldClassifierDecoder decodes the ResearchFieldClassifier
    service model's output to a user-friendly one.
    """

    def __init__(self, label_dict):
        """

        :param label_dict: Classifier label mapping.
        :type label_dict: Dict[str, int]
        """
        super().__init__()

        self.label_dict = {index: label for label, index in label_dict.items()}

    @overrides(check_signature=False)
    def decode(
        self, model_output: Union[Any, Generator[Any, None, None]], top_n: int, **kwargs: Any
    ) -> Any:
        logits = model_output["logits"]
        top_n_scores, top_n_indices = torch.topk(logits, k=top_n)
        top_n_predicts = [
            {"research_field": self.label_dict[indices.item()], "score": score.item()}
            for score, indices in zip(top_n_scores[0], top_n_indices[0])
        ]
        return top_n_predicts
