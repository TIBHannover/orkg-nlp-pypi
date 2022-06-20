""" TDM-Extraction service decoder. """
from ctypes import Union
from typing import Any, Dict, Tuple, Generator, List, Iterable

import numpy as np
import torch
from overrides import overrides
from pandas import DataFrame

from orkgnlp.common.service.base import ORKGNLPBaseDecoder


class TdmExtractorDecoder(ORKGNLPBaseDecoder):
    """
    The TdmExtractorDecoder decodes the TDM-Extraction service model's output
    to a user-friendly one.
    """

    def __init__(self, labels: DataFrame):
        """

        :param labels: TDM gold labels given as one-columned-dataframe
        """
        super().__init__()

        self.labels: DataFrame = labels

    @overrides(check_signature=False)
    def decode(
            self,
            model_output: Generator[Any, None, None],
            top_n: int,
            **kwargs: Any
    ) -> Any:
        self.labels['prob'] = np.NaN

        for batch_idx, batch in enumerate(model_output):
            predictions = torch.sigmoid(batch.logits)

            for predictions_idx, (true, false) in enumerate(predictions):
                if true.item() > false.item():
                    label_index = batch_idx * predictions.shape[0] + predictions_idx
                    self.labels.at[label_index, 'prob'] = true.item()

        candidates = self.labels[self.labels['prob'].notnull()]
        candidates = candidates.sort_values(by='prob', ascending=False)

        return self._prepare_service_output(
            candidates[0][:top_n].tolist(),
            candidates['prob'][:top_n].tolist()
        )

    @staticmethod
    def _prepare_service_output(tdms: List[str], scores: List[float]) -> List[Dict[str, Any]]:
        service_output = []

        for i, tdm in enumerate(tdms):
            t, d, m = tdm.split('#')

            service_output.append({
                'task': t,
                'dataset': d,
                'metric': m,
                'score': scores[i]
            })

        return service_output
