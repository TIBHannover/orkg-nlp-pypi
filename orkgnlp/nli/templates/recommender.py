""" Templates Recommendation service. """
from typing import Any
from transformers import BertForSequenceClassification

from orkgnlp.common.service.base import ORKGNLPBaseService, ORKGNLPBaseEncoder, ORKGNLPBaseRunner, ORKGNLPBaseDecoder
from orkgnlp.common.service.runners import ORKGNLPTorchRunner
from orkgnlp.common.util import io
from orkgnlp.nli.templates.decoder import TemplatesRecommenderDecoder
from orkgnlp.nli.templates.encoder import TemplatesRecommenderEncoder


class TemplatesRecommender(ORKGNLPBaseService):
    """
    The TemplatesRecommender requires a transformers.BertForSequenceClassification pretrained model and the templates
    used for training as premises.
    The required files are downloaded while initiation, if it has not happened before.

    You can pass the parameter ``force_download=True`` to remove and re-download the previous downloaded service files.
    """
    SERVICE_NAME = 'templates-recommendation'

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(self.SERVICE_NAME, *args, **kwargs)
        templates = io.read_json(self._config.requirements['labels'])['templates']

        encoder: ORKGNLPBaseEncoder = TemplatesRecommenderEncoder(templates)
        runner: ORKGNLPBaseRunner = ORKGNLPTorchRunner(
            io.load_transformers_pretrained(self._config.service_dir, BertForSequenceClassification)
        )
        decoder: ORKGNLPBaseDecoder = TemplatesRecommenderDecoder(templates)

        self._register_pipeline('main', encoder, runner, decoder)

    def __call__(self, title: str, abstract: str, top_n: int = 5):
        """
        Recommends templates for a research paper.

        :param title: Title of the research paper.
        :param abstract: Abstract of the research paper.
        :param top_n: Top n results to be extracted. Defaults to 5.
        :return: List of templates.
        """
        return self._run(
            raw_input='{} {}'.format(title, abstract),
            top_n=top_n,
            multiple_batches=True
        )
