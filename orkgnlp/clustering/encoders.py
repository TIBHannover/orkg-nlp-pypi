""" Common encoders for the clustering services. """
from overrides import overrides

from orkgnlp.common.util import text
from orkgnlp.common.service.base import ORKGNLPBaseEncoder
from orkgnlp.common.service.runners import ORKGNLPONNXRunner


class TfidfKmeansEncoder(ORKGNLPBaseEncoder):
    """
    The TfidfKmeansEncoder encodes the given input to a TF-IDF vector
    needed to execute a Kmeans onnx model.
    """

    def __init__(self, vectorizer):
        """

        :param vectorizer: The TF-IDF vectorizer needed for the encoding.
        :type vectorizer: Loaded ``onnx`` object.
        """
        super().__init__()
        self._vectorizer = ORKGNLPONNXRunner(vectorizer)

    @overrides
    def encode(self, raw_input, **kwargs):
        preprocessed_text = self._text_process(raw_input)
        output, _ = self._vectorizer.run(
            inputs=([preprocessed_text],),
            output_names=['variable']
        )
        return (output[0][0], ), kwargs

    @staticmethod
    def _text_process(q):
        q = text.remove_punctuation(q)
        q = text.remove_stopwords(q)
        q = text.lemmatize(q)
        return q.lower()
