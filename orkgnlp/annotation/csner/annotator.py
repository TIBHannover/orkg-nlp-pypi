""" CS-NER service. """
from orkgnlp.annotation.csner._annotator_config import config
from orkgnlp.annotation.csner._ncrfpp import evaluation
from orkgnlp.annotation.csner._ncrfpp.model.seqlabel import SeqLabel
from orkgnlp.annotation.csner._ncrfpp.utils.data import Data
from orkgnlp.common.base import ORKGNLPBase
from orkgnlp.common.util import io
from orkgnlp.common.util.decorators import singleton


class CSNer(ORKGNLPBase):
    """
    The CSNer follows the singleton pattern, i.e. only one instance can be obtained from it.

    It requires abstracts and titles models and their configurations obtained during the training.
    The required files are downloaded while initiation, if it has not happened before.

    You can pass the parameter ``force_download=True`` to remove and re-download the previous downloaded service files.
    """

    @singleton
    def __new__(cls):
        pass

    def __init__(self, *args, **kwargs):
        super().__init__(config['service_name'], *args, **kwargs)

        self._titles_data = self._create_data(config['paths']['titles_dset'])
        self._titles_model = self._create_model(self._titles_data, config['paths']['titles_model'])

        self._abstracts_data = self._create_data(config['paths']['abstracts_dset'])
        self._abstracts_model = self._create_model(self._abstracts_data, config['paths']['abstracts_model'])

    def annotate_title(self, title):
        """
        Applies Named Entity Recognition on the given paper's ``title``.

        :param title: Paper's title.
        :type title: str
        :return: A dict representing the annotated parts of the given ``title``.
        """
        return self._annotate(q=title, data=self._titles_data, model=self._titles_model)

    def annotate_abstract(self, abstract):
        """
        Applies Named Entity Recognition on the given paper's ``abstract``.

        :param abstract: Paper's abstract.
        :type abstract: str
        :return: A dict representing the annotated parts of the given ``abstract``.
        """
        return self._annotate(q=abstract, data=self._abstracts_data, model=self._abstracts_model)

    def annotate(self, title, abstract):
        """
        Applies Named Entity Recognition on each of the given paper's ``title`` and ``abstract``.

        :param title: Paper's title.
        :type title: str
        :param abstract: Paper's abstract.
        :type abstract: str
        :return: A dict representing the annotated parts for each of the given ``title`` and ``abstract``.
        """
        return {
            'title': self.annotate_title(title),
            'abstract': self.annotate_abstract(abstract)
        }

    def _annotate(self, q, data, model):
        data.generate_instance(q)
        results = evaluation.predict(data, model)
        entities = data.get_entities(results)
        return self._prepare_annotations(entities)

    @staticmethod
    def _prepare_annotations(entities):
        annotations = []

        for concept in entities:
            annotations.append({
                'concept': concept,
                'entities': entities[concept]
            })

        return annotations

    @staticmethod
    def _create_model(data, model_path):
        model = SeqLabel(data)
        model.load_state_dict(io.load_torch(model_path))
        return model

    @staticmethod
    def _create_data(dset_path):
        data = Data()
        data.load(io.read_pickle(dset_path))
        return data
