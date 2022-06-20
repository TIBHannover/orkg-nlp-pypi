""" TDM-Extraction service. """
from typing import Any

from pandas import DataFrame
from transformers import XLNetForSequenceClassification

from orkgnlp.annotation.tdm.decoder import TdmExtractorDecoder
from orkgnlp.annotation.tdm.encoder import TdmExtractorEncoder
from orkgnlp.common.util import io

from orkgnlp.common.service.base import ORKGNLPBaseService
from orkgnlp.annotation.tdm._config import config
from orkgnlp.common.service.runners import ORKGNLPTorchRunner


class TdmExtractor(ORKGNLPBaseService):
    """
    The TdmExtractor requires a transformers.XLNetForSequenceClassification pretrained model and a TDM gold labels file.
    The required files are downloaded while initiation, if it has not happened before.

    You can pass the parameter ``force_download=True`` to remove and re-download the previous downloaded service files.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(config['service_name'], *args, **kwargs)

        labels: DataFrame = io.read_csv(config['paths']['labels'], sep='\t')

        if self._unittest:
            labels = labels.head()

        encoder: TdmExtractorEncoder = TdmExtractorEncoder(labels, self._batch_size)
        runner: ORKGNLPTorchRunner = ORKGNLPTorchRunner(
            io.load_transformers_pretrained(config['paths']['model_dir'], XLNetForSequenceClassification)
        )
        decoder: TdmExtractorDecoder = TdmExtractorDecoder(labels)
        self._register_pipeline('main', encoder, runner, decoder)

    def __call__(self, text: str, top_n: int = 5) -> Any:
        """
        Extracts Task-Dataset-Metric (TDM) entities from a given
        DocTAET (Title, Abstract, ExperimentalSetup, TableInfo) ``text``

        :param text: `DocTAET <https://doi.org/10.1007/978-3-030-91669-5_35>`_ represented text.
        :param top_n: Top n results to be extracted. Defaults to 5.
        :return: A list of TDMs.
        """
        return self._run(
            raw_input=text,
            top_n=top_n,
            multiple_batches=True,
        )
