""" Predicates recommendation service decoder. """
from typing import Union, Tuple, Dict, Any, Generator, List

import numpy as np
from overrides import overrides
from pandas import DataFrame

from orkgnlp.common.service.base import ORKGNLPBaseDecoder


class PredicatesRecommenderDecoder(ORKGNLPBaseDecoder):
    """
    The PredicatesRecommenderDecoder decodes the Predicates' recommendation service model's output
    to a user-friendly one.
    """

    def __init__(self, train_df: DataFrame, predicates: Dict[str, List[Dict[str, str]]]):
        """

        :param train_df: The training dataframe of the service.
        :param predicates: Dict object representing the mapping from comparisons to predicates.
        """
        super().__init__()
        self._train_df = train_df
        self._predicates = predicates

    @overrides
    def decode(
            self,
            model_output: Union[Any, Generator[Any, None, None]],
            **kwargs: Any
    ) -> Any:
        cluster_label, model_labels_ = model_output[0], model_output[1]
        cluster_instances_indices = np.argwhere(cluster_label == model_labels_).squeeze(1)
        cluster_instances = self._train_df.iloc[cluster_instances_indices]
        comparison_ids = cluster_instances['comparison_id'].unique()
        return self._map_to_predicates(comparison_ids)

    def _map_to_predicates(self, comparison_ids: List[str]) -> List[Dict[str, str]]:
        predicates = []

        for comparison_id in comparison_ids:
            predicates.extend(self._predicates[comparison_id])

        return predicates
