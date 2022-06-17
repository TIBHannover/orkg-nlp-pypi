from unittest import TestCase

from orkgnlp.clustering import BioassaysSemantifier


class TestBioassaysSemantifier(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.semantifier = BioassaysSemantifier(force_download=False)

    @classmethod
    def tearDownClass(cls):
        del cls

    def test_singleton(self):
        another_semantifier = BioassaysSemantifier()
        self.assertEqual(self.semantifier, another_semantifier)

    def test_semantify(self):
        text = 'long text'
        self.assertIsInstance(self.semantifier(text=text), list)
