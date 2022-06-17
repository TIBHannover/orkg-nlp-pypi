from unittest import TestCase

from orkgnlp.annotation import TdmExtractor


class TestCSNer(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.extractor = TdmExtractor(force_download=False, _unittest=True)
        cls.text = 'short'

    @classmethod
    def tearDownClass(cls):
        del cls

    def test_singleton(self):
        another_annotator = TdmExtractor()
        self.assertEqual(self.extractor, another_annotator)

    def test_extract_tdms(self):
        tdms = self.extractor(text=self.text)

        self.assertIsInstance(tdms, list)
        for tdm in tdms:
            self.assertIsInstance(tdm, dict)
            self.assertIn('task', tdm)
            self.assertIn('dataset', tdm)
            self.assertIn('metric', tdm)
            self.assertIn('score', tdm)

