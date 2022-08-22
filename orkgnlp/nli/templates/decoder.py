""" Templates Recommendation service decoder. """
from typing import Dict, Generator, Any, List
from overrides import overrides

from orkgnlp.common.service.base import ORKGNLPBaseDecoder


class TemplatesRecommenderDecoder(ORKGNLPBaseDecoder):
    """
    The TemplatesRecommenderDecoder decodes the Templates Recommendation service model's output
    to a user-friendly one.
    """

    def __init__(self, templates: List[Dict[str, str]]):
        """

        :param templates: templates used for training the service models as premises.
        """
        super().__init__()

        self.templates: List[Dict[str, str]] = templates
        self.id2label = {
            '0': 'entailment',
            '1': 'contradiction',
            '2': 'neutral'
        }

    @overrides(check_signature=False)
    def decode(
            self,
            model_output: Generator[Any, None, None],
            top_n: int,
            **kwargs: Any
    ) -> Any:

        templates = []
        for idx, prediction in enumerate(model_output):
            class_id = prediction.logits.argmax(dim=-1).item()
            score = prediction.logits[0][class_id].item()

            if self.id2label[str(class_id)] == 'entailment':
                templates.append({
                    'id': self.templates[idx]['id'],
                    'label': self.templates[idx]['label'],
                    'score': score,
                })

        templates = sorted(templates, key=lambda i: i['score'], reverse=True)

        return templates[:top_n]
