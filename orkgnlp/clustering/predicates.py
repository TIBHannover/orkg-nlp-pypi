""" Predicates recommendation service. """
import numpy as np
import onnxruntime as rt

from orkgnlp.clustering._predicates_config import config
from orkgnlp.common.base import ORKGNLPBase
from orkgnlp.common.util import io, text
from orkgnlp.common.util.decorators import singleton


class PredicatesRecommender(ORKGNLPBase):
    """
    The PredicatesRecommender follows the singleton pattern, i.e. only one instance can be obtained from it.

    It requires a clustering model, vectorizer, training set and predicates. The required files are downloaded while
    initiation, if it has not happened before.
    """

    @singleton
    def __new__(cls):
        pass

    def __init__(self, *args, **kwargs):
        super().__init__(config['service_name'], *args, **kwargs)

        self._model = io.read_onnx(config['paths']['model'])
        self._vectorizer = io.read_onnx(config['paths']['vectorizer'])
        self._train_df = io.read_df_from_json(config['paths']['training_data'], key='instances')
        self._predicates = io.read_json(config['paths']['mapping'])

    def recommend(self, title, abstract):
        """
        Recommends predicates for a research paper.

        :param title: Title of the research paper.
        :type title: str
        :param abstract: Abstract of the research paper.
        :type abstract: str
        :return: List of predicates.
        """
        return self._recommend(q='{} {}'.format(title, abstract))

    def _recommend(self, q):
        vectorized_text = self._vectorize_input(q)

        comparison_ids = self._predict_comparisons(vectorized_text)
        return self._map_to_predicates(comparison_ids)

    def _vectorize_input(self, q):
        preprocessed_text = self._text_process(q)
        session = rt.InferenceSession(self._vectorizer.SerializeToString())
        output = session.run(['variable'], {session.get_inputs()[0].name: [[preprocessed_text]]})
        return output[0][0]

    def _predict_comparisons(self, vectorized_text):
        session = rt.InferenceSession(self._model.SerializeToString())
        output = session.run(['label', 'labels_'], {session.get_inputs()[0].name: [vectorized_text]})
        cluster_label, model_labels_ = output[0], output[1]
        cluster_instances_indices = np.argwhere(cluster_label == model_labels_).squeeze(1)
        cluster_instances = self._train_df.iloc[cluster_instances_indices]
        comparison_ids = cluster_instances['comparison_id'].unique()
        return comparison_ids

    def _map_to_predicates(self, comparison_ids):
        predicate_ids = []
        predicates = []

        for comparison_id in comparison_ids:
            for predicate in self._predicates[comparison_id]:
                if predicate['id'] in predicate_ids:
                    continue

                predicate_ids.append(predicate['id'])
                predicates.append(predicate)

        return predicates

    @staticmethod
    def _text_process(q):
        q = text.remove_punctuation(q)
        q = text.remove_stopwords(q)
        q = text.lemmatize(q)
        return q.lower()
