from unittest import TestCase

from orkgnlp.annotation import CSNer


class TestCSNer(TestCase):

    def setUp(self):
        self.annotator = CSNer(force_download=False)
        self.title = 'Open Research Knowledge Graph: Next Generation Infrastructure for Semantic Scholarly Knowledge'
        self.abstract = 'Despite improved digital access to scholarly knowledge in recent decades, scholarly communication remains exclusively document-based. In this form, scholarly knowledge is hard to process automatically. We present the first steps towards a knowledge graph based infrastructure that acquires scholarly knowledge in machine actionable form thus enabling new possibilities for scholarly knowledge curation, publication and processing. The primary contribution is to present, evaluate and discuss multi-modal scholarly knowledge acquisition, combining crowdsourced and automated techniques. We present the results of the first user evaluation of the infrastructure with the participants of a recent international conference. Results suggest that users were intrigued by the novelty of the proposed infrastructure and by the possibilities for innovative scholarly knowledge processing it could enable.'

    def test_singleton(self):
        another_annotator = CSNer()
        self.assertEqual(self.annotator, another_annotator)

    def test_annotate_title(self):
        annotations = self.annotator.annotate_title(title=self.title)

        self.assertIsInstance(annotations, dict)
        for key in annotations:
            self.assertIsInstance(annotations[key], list)

    def test_annotate_abstract(self):
        annotations = self.annotator.annotate_abstract(abstract=self.abstract)

        self.assertIsInstance(annotations, dict)
        for key in annotations:
            self.assertIsInstance(annotations[key], list)

    def test_annotate(self):
        result = self.annotator.annotate(title=self.title, abstract=self.abstract)

        self.assertIsInstance(result, dict)
        self.assertEqual(2, len(result))

        for annotations in result:
            self.assertIsInstance(result[annotations], dict)

            for key in result[annotations]:
                self.assertIsInstance(result[annotations][key], list)
