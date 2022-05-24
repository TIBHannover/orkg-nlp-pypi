from unittest import TestCase
from orkgnlp.common.util import text


class TestText(TestCase):

    def test_remove_punctuation(self):
        s = '!.@$%&/()=?'
        self.assertFalse(text.remove_punctuation(s))

    def test_remove_stopwords(self):
        s = 'hello the was'
        self.assertEqual('hello', text.remove_stopwords(s))
