""" Agri-NER service. """

from typing import Any

from transformers import pipeline

from orkgnlp.annotation.agriner.decoder import AgriNerDecoder
from orkgnlp.common.service.base import ORKGNLPBaseService, ORKGNLPBaseRunner, ORKGNLPBaseEncoder, ORKGNLPBaseDecoder
from orkgnlp.common.service.runners import ORKGNLPTorchRunner


class AgriNer(ORKGNLPBaseService):
    """
    The AgriNer requires a classification model trained on titles and its configurations obtained during the training.
    The required files are downloaded while initiation, if it has not happened before.

    You can pass the parameter ``force_download=True`` to remove and re-download the previous downloaded service files.
    """
    SERVICE_NAME = 'agri-ner'

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(self.SERVICE_NAME, *args, **kwargs)

        encoder: ORKGNLPBaseEncoder = ORKGNLPBaseEncoder()
        _model = pipeline(
            task='token-classification',
            model=self._config.service_dir,
            tokenizer='bert-base-cased',
            aggregation_strategy='simple'
        )
        runner: ORKGNLPBaseRunner = ORKGNLPTorchRunner(_model)
        decoder: ORKGNLPBaseDecoder = AgriNerDecoder()
        self._register_pipeline('main', encoder, runner, decoder)

    def __call__(self, title: str) -> Any:
        """
        Applies Named Entity Recognition on the given paper's title.

        :param title: Paper's title.
        :return: A list of the annotated parts for the given text is returned.
        """
        return self._run(
            raw_input=title
        )
