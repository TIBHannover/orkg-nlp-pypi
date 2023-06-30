# -*- coding: utf-8 -*-
""" ResearchFieldClassifier service. """
from typing import Any

from orkgnlp.annotation.rfclf.decoder import ResearchFieldClassifierDecoder
from orkgnlp.annotation.rfclf.encoder import ResearchFieldClassifierEncoder
from orkgnlp.common.config import orkgnlp_context
from orkgnlp.common.service.base import (
    ORKGNLPBaseDecoder,
    ORKGNLPBaseEncoder,
    ORKGNLPBaseRunner,
    ORKGNLPBaseService,
)
from orkgnlp.common.service.runners import ORKGNLPTorchRunner
from orkgnlp.common.util import io
from orkgnlp.common.util.exceptions import ORKGNLPValidationError


class ResearchFieldClassifier(ORKGNLPBaseService):
    """
    The ResearchFieldClassifier requires a torch-script model trained on abstracts and
    a label dictionary file containing research fields and corresponding indices.
    The required files are downloaded while initiation, if it has not happened before

    You can pass the parameter ``force_download=True`` to remove and re-download the previous downloaded service files.
    """

    SERVICE_NAME = "research-fields-classification"

    def __init__(self, *args, **kwargs):
        super().__init__(self.SERVICE_NAME, *args, **kwargs)
        requirements = self._config.requirements

        encoder: ORKGNLPBaseEncoder = ResearchFieldClassifierEncoder()
        runner: ORKGNLPBaseRunner = ORKGNLPTorchRunner(io.load_torch_jit(requirements["model"]))
        decoder: ORKGNLPBaseDecoder = ResearchFieldClassifierDecoder(
            io.read_json(requirements["label_dict"])
        )

        self._register_pipeline("main", encoder, runner, decoder)

    def __call__(self, raw_input: str, top_n: int = 5) -> Any:
        """
        Classifies a paper into a research field by processing the given abstract.

        :param raw_input: Combined paper's title and abstract
        :param top_n: The top n research fields to be retrieved
        :return: A list of n most likely research fields for the paper
        :raise orkgnlp.common.util.exceptions.ORKGNLPValidationError: If no abstract is given
        """
        if not raw_input:
            raise ORKGNLPValidationError("Abstract must be provided")

        return self._run(raw_input=raw_input, top_n=top_n)


orkgnlp_context.get("SERVICE_MAP")[ResearchFieldClassifier.SERVICE_NAME] = ResearchFieldClassifier
