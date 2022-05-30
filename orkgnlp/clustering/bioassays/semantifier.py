""" BioAssays semantification service. """

import onnxruntime as rt

from orkgnlp.clustering.bioassays.decoder import BioassaysSemantifierDecoder
from orkgnlp.clustering.encoders import TfidfKmeansEncoder
from orkgnlp.common.base import ORKGNLPBaseService
from orkgnlp.common.runners import ORKGNLPONNXRunner
from orkgnlp.common.util import io, text
from orkgnlp.common.util.decorators import singleton
from orkgnlp.clustering.bioassays._config import config


class BioassaysSemantifier(ORKGNLPBaseService):
    """
    The BioassaysSemantifier follows the singleton pattern, i.e. only one instance can be obtained from it.

    It requires a clustering model, vectorizer and mapping. The required files are downloaded while
    initiation, if it has not happened before.

    You can pass the parameter ``force_download=True`` to remove and re-download the previous downloaded service files.
    """

    @singleton
    def __new__(cls):
        pass

    def __init__(self, *args, **kwargs):
        super().__init__(config['service_name'], *args, **kwargs)

        self._encoder = TfidfKmeansEncoder(io.read_onnx(config['paths']['vectorizer']))
        self._runner = ORKGNLPONNXRunner(io.read_onnx(config['paths']['model']))
        self._decoder = BioassaysSemantifierDecoder(io.read_json(config['paths']['mapping']))

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
