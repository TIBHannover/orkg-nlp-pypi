import numpy as np
import torch
from overrides import overrides

from orkgnlp.common.service.base import ORKGNLPBaseDecoder


class TdmExtractorDecoder(ORKGNLPBaseDecoder):

    def __init__(self, labels):
        super().__init__()

        self.labels = labels

    @overrides(check_signature=False)
    def decode(self, model_output, top_n, **kwargs):
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
    def _prepare_service_output(tdms, scores):
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
