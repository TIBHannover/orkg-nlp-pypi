""" Predicates recommendation service. """
from typing import Any, Dict

from orkgnlp.clustering.predicates._config import config
from orkgnlp.clustering.encoders import TfidfKmeansEncoder
from orkgnlp.clustering.predicates.decoder import PredicatesRecommenderDecoder
from orkgnlp.common.service.base import ORKGNLPBaseService
from orkgnlp.common.service.runners import ORKGNLPONNXRunner
from orkgnlp.common.util import io


class PredicatesRecommender(ORKGNLPBaseService):
    """
    The PredicatesRecommender requires a clustering model, vectorizer, training set and predicates.
    The required files are downloaded while initiation, if it has not happened before.

    You can pass the parameter ``force_download=True`` to remove and re-download the previous downloaded service files.
    """
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(config['service_name'], *args, **kwargs)

        encoder: TfidfKmeansEncoder = TfidfKmeansEncoder(io.read_onnx(config['paths']['vectorizer']))
        runner: ORKGNLPONNXRunner = ORKGNLPONNXRunner(io.read_onnx(config['paths']['model']))
        decoder: PredicatesRecommenderDecoder = PredicatesRecommenderDecoder(
            io.read_df_from_json(config['paths']['training_data'], key='instances'),
            io.read_json(config['paths']['mapping'])
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
