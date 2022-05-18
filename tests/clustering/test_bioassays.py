from unittest import TestCase

from orkgnlp.clustering import BioassaysSemantifier


class TestBioassaysSemantifier(TestCase):

    def setUp(self):
        self.semantifier = BioassaysSemantifier(force_download=False)

    def test_singleton(self):
        another_semantifier = BioassaysSemantifier()
        self.assertEqual(self.semantifier, another_semantifier)

    def test_semantify(self):
        text = 'long text'
        self.assertIsInstance(self.semantifier.semantify(text=text), list)
