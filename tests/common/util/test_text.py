# -*- coding: utf-8 -*-
from unittest import TestCase

from orkgnlp.common.util import text


class TestText(TestCase):
    def test_remove_punctuation(self):
        s = "!.@$%&/()=?"
        self.assertFalse(text.remove_punctuation(s))

    def test_remove_stopwords(self):
        s = "hello the was"
        self.assertEqual("hello", text.remove_stopwords(s))

    def test_replace(self):
        s = "hello_ wo.rld-"
        self.assertEqual("hello  wo rld ", text.replace(s, [r"\s+-\s+", "-", "_", r"\."], " "))

    def test_trim(self):
        s = "hello     world "
        self.assertEqual("hello world", text.trim(s))
