""" BioAssays semantification service. """

from orkgnlp.clustering.bioassays.decoder import BioassaysSemantifierDecoder
from orkgnlp.clustering.encoders import TfidfKmeansEncoder
from orkgnlp.common.service.base import ORKGNLPBaseService
from orkgnlp.common.service.runners import ORKGNLPONNXRunner
from orkgnlp.common.util import io
from orkgnlp.clustering.bioassays._config import config


class BioassaysSemantifier(ORKGNLPBaseService):
    """
    The BioassaysSemantifier  requires a clustering model, vectorizer and mapping.
    The required files are downloaded while initiation, if it has not happened before.

    You can pass the parameter ``force_download=True`` to remove and re-download the previous downloaded service files.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(config['service_name'], *args, **kwargs)

        encoder = TfidfKmeansEncoder(io.read_onnx(config['paths']['vectorizer']))
        runner = ORKGNLPONNXRunner(io.read_onnx(config['paths']['model']))
        decoder = BioassaysSemantifierDecoder(io.read_json(config['paths']['mapping']))

        self._register_pipeline('main', encoder, runner, decoder)

    def __call__(self, text):
        """
        Semantifies a given BioAssay's description text.

        :param text: BioAssay's text to be semantified.
        :type text: str
        :return: Dictionary object of semantified properties, resources and labels.
        """

        return self._run(
            raw_input=text
        )
