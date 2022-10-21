""" Predicates recommendation service. """
from typing import Any, Dict

from orkgnlp.clustering.encoders import TransformerKmeansEncoder
from orkgnlp.clustering.predicates.decoder import PredicatesRecommenderDecoder
from orkgnlp.common.service.base import ORKGNLPBaseService, ORKGNLPBaseEncoder, ORKGNLPBaseRunner, ORKGNLPBaseDecoder
from orkgnlp.common.service.runners import ORKGNLPONNXRunner
from orkgnlp.common.util import io


class PredicatesRecommender(ORKGNLPBaseService):
    """
    The PredicatesRecommender requires a clustering model, vectorizer, training set and predicates.
    The required files are downloaded while initiation, if it has not happened before.

    You can pass the parameter ``force_download=True`` to remove and re-download the previous downloaded service files.
    """
    SERVICE_NAME = 'predicates-clustering'

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(self.SERVICE_NAME, *args, **kwargs)
        requirements = self._config.requirements

        encoder: ORKGNLPBaseEncoder = TransformerKmeansEncoder('allenai/scibert_scivocab_uncased')
        runner: ORKGNLPBaseRunner = ORKGNLPONNXRunner(io.read_onnx(requirements['model']))
        decoder: ORKGNLPBaseDecoder = PredicatesRecommenderDecoder(
            io.read_df_from_json(requirements['training_data'], key='instances'),
            io.read_json(requirements['mapping'])
        )
        self._register_pipeline('main', encoder, runner, decoder)

    def __call__(self, title: str, abstract: str):
        """
        Recommends predicates for a research paper.

        :param title: Title of the research paper.
        :param abstract: Abstract of the research paper.
        :return: List of predicates.
        """
        return self._run(
            raw_input='{} {}'.format(title, abstract),
            output_names=['label', 'labels_']
        )
