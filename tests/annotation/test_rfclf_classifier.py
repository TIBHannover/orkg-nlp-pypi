# -*- coding: utf-8 -*-
from unittest import TestCase

from orkgnlp.annotation import ResearchFieldClassifier


class TestResearchFieldClassifier(TestCase):
    def setUp(self):
        self.classifier = ResearchFieldClassifier(force_download=False)
        self.abstract = """Understanding of cybersecurity threat landscape especially information about threat
                           actor is a challenging task as these information are usually hidden and scattered.
                           The online news had became one of the popular and important source of information for
                           cybersecurity personnels to understand about the activities conducted by these threat
                           actors. In this paper, we propose a framework to create knowledge graph of threat actor
                           by building ontology of threat actor and named entity recognition system to extract
                           cybersecurity-related entities. The resulting ontology and model can be used to
                           automatically extract cybesecurity-related entities from an article and create knowledge
                           graph of threatactor."""
        self.addCleanup(self.classifier.release_memory)

    def test_singleton(self):
        another_classifier = ResearchFieldClassifier()
        self.assertEqual(self.classifier, another_classifier)

    def test_classify(self):
        top_n = 6
        labels = self.classifier(abstract=self.abstract, top_n=top_n)

        self.assertIsInstance(labels, list)
        self.assertEqual(len(labels), top_n)
