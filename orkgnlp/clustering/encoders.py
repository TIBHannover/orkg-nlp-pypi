""" Common encoders for the clustering services. """
from typing import Any, Dict, Tuple

from onnx import ModelProto
from overrides import overrides
from sentence_transformers import SentenceTransformer

from orkgnlp.common.util import text
from orkgnlp.common.service.base import ORKGNLPBaseEncoder
from orkgnlp.common.service.runners import ORKGNLPONNXRunner


class TfidfKmeansEncoder(ORKGNLPBaseEncoder):
    """
    The TfidfKmeansEncoder encodes the given input to a TF-IDF vector
    needed to execute a Kmeans onnx model.
    """

    def __init__(self, vectorizer: ModelProto):
        """

        :param vectorizer: The TF-IDF vectorizer needed for the encoding.
        """
        super().__init__()
        self._vectorizer: ORKGNLPONNXRunner = ORKGNLPONNXRunner(vectorizer)

    @overrides
    def encode(self, raw_input: Any, **kwargs: Any) -> Tuple[Any, Dict[str, Any]]:
        preprocessed_text = self._text_process(raw_input)
        output, _ = self._vectorizer.run(
            inputs=([preprocessed_text],),
            output_names=['variable']
        )
        return (output[0][0], ), kwargs

    @staticmethod
    def _text_process(q: str) -> str:
        q = text.remove_punctuation(q)
        q = text.remove_stopwords(q)
        q = text.lemmatize(q)
        return q.lower()


class TransformerKmeansEncoder(ORKGNLPBaseEncoder):
    """
    The SciBERTKmeansEncoder encodes the given input to a SciBERT vector
    needed to execute a Kmeans onnx model.
    """

    def __init__(self, transformer_path: str):
        """

        :param transformer_path: Path to transformers model. Can be a model name on Huggingface.
        """
        super().__init__()
        self._vectorizer: SentenceTransformer = SentenceTransformer(transformer_path)

    @overrides
    def encode(self, raw_input: Any, **kwargs: Any) -> Tuple[Any, Dict[str, Any]]:
        return (self._vectorizer.encode([raw_input])[0], ), kwargs
