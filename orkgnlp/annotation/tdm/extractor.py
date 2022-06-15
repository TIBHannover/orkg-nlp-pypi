from transformers import XLNetForSequenceClassification

from orkgnlp.annotation.tdm.decoder import TdmExtractorDecoder
from orkgnlp.annotation.tdm.encoder import TdmExtractorEncoder
from orkgnlp.common.util import io

from orkgnlp.common.service.base import ORKGNLPBaseService
from orkgnlp.annotation.tdm._config import config
from orkgnlp.common.service.runners import ORKGNLPTorchRunner


class TdmExtractor(ORKGNLPBaseService):

    def __init__(self, *args, **kwargs):
        super().__init__(config['service_name'], *args, **kwargs)

        labels = io.read_csv(config['paths']['labels'], sep='\t')
        encoder = TdmExtractorEncoder(labels)
        runner = ORKGNLPTorchRunner(
            io.load_transformers_pretrained(config['paths']['model_dir'], XLNetForSequenceClassification)
        )
        decoder = TdmExtractorDecoder(labels)
        self._register_pipeline('main', encoder, runner, decoder)

    def __call__(self, text, top_n=5):
        """
        TODO:

        :param text:
        :return:
        """
        return self._run(
            raw_input=text,
            top_n=top_n,
            multiple_batches=True,
            logits=True
        )
