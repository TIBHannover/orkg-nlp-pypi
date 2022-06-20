from unittest import TestCase

from orkgnlp.clustering import BioassaysSemantifier


class TestBioassaysSemantifier(TestCase):

    def setUp(self):
        self.semantifier = BioassaysSemantifier(force_download=False)
        self.addCleanup(self.semantifier._release_memory)

    def test_singleton(self):
        another_semantifier = BioassaysSemantifier()
        self.assertEqual(self.semantifier, another_semantifier)

    def test_semantify(self):
        text = 'long text'
        self.assertIsInstance(self.semantifier(text=text), list)
