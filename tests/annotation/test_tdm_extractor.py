# -*- coding: utf-8 -*-
from unittest import TestCase

from orkgnlp.annotation import TdmExtractor


class TestTdmExtractor(TestCase):
    def setUp(self):
        self.extractor = TdmExtractor(force_download=False, _unittest=True)
        self.text = "short"
        self.addCleanup(self.extractor.release_memory)

    def test_singleton(self):
        another_annotator = TdmExtractor()
        self.assertEqual(self.extractor, another_annotator)

    def test_extract_tdms(self):
        tdms = self.extractor(text=self.text)

        self.assertIsInstance(tdms, list)
        for tdm in tdms:
            self.assertIsInstance(tdm, dict)
            self.assertIn("task", tdm)
            self.assertIn("dataset", tdm)
            self.assertIn("metric", tdm)
            self.assertIn("score", tdm)
