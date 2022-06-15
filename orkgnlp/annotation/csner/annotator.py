""" CS-NER service. """
from orkgnlp.annotation.csner._config import config
from orkgnlp.annotation.csner.decoder import CSNerDecoder
from orkgnlp.annotation.csner.encoder import CSNerEncoder
from orkgnlp.common.service.base import ORKGNLPBaseService
from orkgnlp.common.service.runners import ORKGNLPTorchRunner
from orkgnlp.common.util import io
from orkgnlp.common.util.exceptions import ORKGNLPValidationError


class CSNer(ORKGNLPBaseService):
    """
    The CSNer requires abstracts and titles models and their configurations obtained during the training.
    The required files are downloaded while initiation, if it has not happened before.

    You can pass the parameter ``force_download=True`` to remove and re-download the previous downloaded service files.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(config['service_name'], *args, **kwargs)

        titles_encoder = CSNerEncoder(io.read_json(config['paths']['titles_alphabet']))
        titles_runner = ORKGNLPTorchRunner(io.load_torch_jit(config['paths']['titles_model']))
        titles_decoder = CSNerDecoder(io.read_json(config['paths']['titles_alphabet']))
        self._titles_pipeline_name = 'titles'
        self._register_pipeline(self._titles_pipeline_name, titles_encoder, titles_runner, titles_decoder)

        abstracts_encoder = CSNerEncoder(io.read_json(config['paths']['abstracts_alphabet']))
        abstracts_runner = ORKGNLPTorchRunner(io.load_torch_jit(config['paths']['abstracts_model']))
        abstracts_decoder = CSNerDecoder(io.read_json(config['paths']['abstracts_alphabet']))
        self._abstracts_pipeline_name = 'abstracts'
        self._register_pipeline(self._abstracts_pipeline_name, abstracts_encoder, abstracts_runner, abstracts_decoder)

    def __call__(self, title=None, abstract=None):
        """
        Applies Named Entity Recognition on the given paper's ``title`` and/or ``abstract``.

        :param title: Paper's title.
        :type title: str
        :param abstract: Paper's abstract.
        :type abstract: str
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
