""" Predicates recommendation service. """

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
    def __init__(self, *args, **kwargs):
        super().__init__(config['service_name'], *args, **kwargs)

        encoder = TfidfKmeansEncoder(io.read_onnx(config['paths']['vectorizer']))
        runner = ORKGNLPONNXRunner(io.read_onnx(config['paths']['model']))
        decoder = PredicatesRecommenderDecoder(
            io.read_df_from_json(config['paths']['training_data'], key='instances'),
            io.read_json(config['paths']['mapping'])
        )
        self._register_pipeline('main', encoder, runner, decoder)

    def __call__(self, title, abstract):
        """
        Recommends predicates for a research paper.

        :param title: Title of the research paper.
        :type title: str
        :param abstract: Abstract of the research paper.
        :type abstract: str
        :return: List of predicates.
        """
        return self._run(
            raw_input='{} {}'.format(title, abstract),
            output_names=['label', 'labels_']
        )
