""" CS-NER service. """
from typing import Any

from orkgnlp.annotation.csner.decoder import CSNerDecoder
from orkgnlp.annotation.csner.encoder import CSNerEncoder
from orkgnlp.common.service.base import ORKGNLPBaseService, ORKGNLPBaseEncoder, ORKGNLPBaseRunner, ORKGNLPBaseDecoder
from orkgnlp.common.service.runners import ORKGNLPTorchRunner
from orkgnlp.common.util import io
from orkgnlp.common.util.exceptions import ORKGNLPValidationError


class CSNer(ORKGNLPBaseService):
    """
    The CSNer requires abstracts and titles models and their configurations obtained during the training.
    The required files are downloaded while initiation, if it has not happened before.

    You can pass the parameter ``force_download=True`` to remove and re-download the previous downloaded service files.
    """
    SERVICE_NAME = 'cs-ner'

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(self.SERVICE_NAME, *args, **kwargs)
        requirements = self._config.requirements

        titles_encoder: ORKGNLPBaseEncoder = CSNerEncoder(io.read_json(requirements['titles_alphabet']))
        titles_runner: ORKGNLPBaseRunner = ORKGNLPTorchRunner(io.load_torch_jit(requirements['titles_model']))
        titles_decoder: ORKGNLPBaseDecoder = CSNerDecoder(io.read_json(requirements['titles_alphabet']))
        self._titles_pipeline_name = 'titles'
        self._register_pipeline(self._titles_pipeline_name, titles_encoder, titles_runner, titles_decoder)

        abstracts_encoder: ORKGNLPBaseEncoder = CSNerEncoder(io.read_json(requirements['abstracts_alphabet']))
        abstracts_runner: ORKGNLPBaseRunner = ORKGNLPTorchRunner(io.load_torch_jit(requirements['abstracts_model']))
        abstracts_decoder: ORKGNLPBaseDecoder = CSNerDecoder(io.read_json(requirements['abstracts_alphabet']))
        self._abstracts_pipeline_name = 'abstracts'
        self._register_pipeline(self._abstracts_pipeline_name, abstracts_encoder, abstracts_runner, abstracts_decoder)

    def __call__(self, title: str = None, abstract: str = None) -> Any:
        """
        Applies Named Entity Recognition on the given paper's ``title`` and/or ``abstract``.

        :param title: Paper's title.
        :param abstract: Paper's abstract.
        :return: If both are given, a dict representing the annotated parts for each of the given
            ``title`` and ``abstract``. Otherwise, a list of the annotated parts for the given text is returned.
        :raise orkgnlp.common.util.exceptions.ORKGNLPValidationError: If neither of the parameters is given.
        """
        if not (title or abstract):
            raise ORKGNLPValidationError('Either title, abstract or both must be provided.')

        if title and abstract:
            return {
                'title': self._annotate(q=title, pipeline_name=self._titles_pipeline_name),
                'abstract': self._annotate(q=abstract, pipeline_name=self._abstracts_pipeline_name)
            }

        if title:
            return self._annotate(q=title, pipeline_name=self._titles_pipeline_name)

        if abstract:
            return self._annotate(q=abstract, pipeline_name=self._abstracts_pipeline_name)

    def _annotate(self, q, pipeline_name):
        return self._run(
            raw_input=q,
            pipline_executor_name=pipeline_name,
            multiple_batches=True
        )
