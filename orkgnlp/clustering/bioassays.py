""" BioAssays semantification service. """

import onnxruntime as rt

from orkgnlp.util import io, text
from orkgnlp.tools import downloader
from orkgnlp.util.decorators import singleton
from orkgnlp.clustering._bioassays_config import config


@singleton
class BioassaysSemantifier:
    """
    The BioassaysSemantifier follows the singleton pattern, i.e. only one instance can be obtained from it.

    It requires a clustering model, vectorizer and mapping. The required files are downloaded while
    initiation, if it has not happened before.

    :note: This class documentation will not be generated with sphinx :autosummary:, because the @singleton decorator
    returns a function that cannot be detected as a class. TODO: fix this issue!
    """

    def __init__(self):
        downloader.exists_or_download(config['service_name'])

        self._model = io.read_onnx(config['paths']['model'])
        self._vectorizer = io.read_onnx(config['paths']['vectorizer'])
        self._mapping = io.read_json(config['paths']['mapping'])

    def semantify(self, text):
        """

        :param text: BioAssay's text to be semantified.
        :type text: str
        :return: Dictionary object of semantified properties, resources and labels.
        """
        vectorized_text = self._vectorize_input(text)

        cluster_label = self._predict_cluster(vectorized_text)

        return self._mapping[str(cluster_label)]

    def _vectorize_input(self, q):
        preprocessed_text = self._text_process(q)
        session = rt.InferenceSession(self._vectorizer.SerializeToString())
        output = session.run(['variable'], {session.get_inputs()[0].name: [[preprocessed_text]]})
        return output[0][0]

    def _predict_cluster(self, vectorized_text):
        session = rt.InferenceSession(self._model.SerializeToString())
        output = session.run(['label'], {session.get_inputs()[0].name: [vectorized_text]})
        return output[0][0]

    @staticmethod
    def _text_process(q):
        q = text.remove_punctuation(q)
        q = text.remove_stopwords(q)
        q = text.lemmatize(q)
        return q.lower()