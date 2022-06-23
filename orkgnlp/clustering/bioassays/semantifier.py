""" BioAssays semantification service. """
from typing import Any

from orkgnlp.clustering.bioassays.decoder import BioassaysSemantifierDecoder
from orkgnlp.clustering.encoders import TfidfKmeansEncoder
from orkgnlp.common.service.base import ORKGNLPBaseService, ORKGNLPBaseRunner, ORKGNLPBaseEncoder, ORKGNLPBaseDecoder
from orkgnlp.common.service.runners import ORKGNLPONNXRunner
from orkgnlp.common.util import io


class BioassaysSemantifier(ORKGNLPBaseService):
    """
    The BioassaysSemantifier  requires a clustering model, vectorizer and mapping.
    The required files are downloaded while initiation, if it has not happened before.

    You can pass the parameter ``force_download=True`` to remove and re-download the previous downloaded service files.
    """
    SERVICE_NAME = 'bioassays-semantification'

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(self.SERVICE_NAME, *args, **kwargs)
        requirements = self._config.requirements

        encoder: ORKGNLPBaseEncoder = TfidfKmeansEncoder(io.read_onnx(requirements['vectorizer']))
        runner: ORKGNLPBaseRunner = ORKGNLPONNXRunner(io.read_onnx(requirements['model']))
        decoder: ORKGNLPBaseDecoder = BioassaysSemantifierDecoder(io.read_json(requirements['mapping']))

        self._register_pipeline('main', encoder, runner, decoder)

    def __call__(self, text: str) -> any:
        """
        Semantifies a given BioAssay's description text.

        :param text: BioAssay's text to be semantified.
        :return: Dictionary object of semantified properties, resources and labels.
        """

        return self._run(
            raw_input=text
        )
