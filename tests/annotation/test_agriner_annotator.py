# -*- coding: utf-8 -*-
from unittest import TestCase

from orkgnlp.annotation import AgriNer


class TestAgriNer(TestCase):
    def setUp(self):
        self.annotator = AgriNer(force_download=False)
        self.title = (
            "Beyond aerodynamics: The critical roles of the circulatory"
            " and systems in maintaining insect functionality"
        )
        self.addCleanup(self.annotator.release_memory)

    def test_singleton(self):
        another_annotator = AgriNer()
        self.assertEqual(self.annotator, another_annotator)

    def test_annotate(self):
        annotations = self.annotator(title=self.title)

        self.assertIsInstance(annotations, list)
        for annotation in annotations:
            self.assertIsInstance(annotation, dict)
            self.assertIn("concept", annotation)
            self.assertIn("entities", annotation)
