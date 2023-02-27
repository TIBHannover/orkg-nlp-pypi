# -*- coding: utf-8 -*-
from unittest import TestCase

from orkgnlp.common.util.datastructure import StrictDict


class TestStrictDict(TestCase):
    def setUp(self):
        self.strict_dict = StrictDict({"key": "value"})

    def test_strict_dict_raises_error(self):
        self.assertRaises(KeyError, self.strict_dict.__setitem__, "unknown_key", "value")
        self.assertRaises(KeyError, self.strict_dict.update, {"unknown_key": "value"})

    def test_strict_dict_does_not_changes_known_nor_unknown_keys(self):
        self.assertRaises(
            KeyError,
            self.strict_dict.update,
            {"key": "new_value", "unknown_key": "value"},
        )
        self.assertEqual("value", self.strict_dict["key"])

    def test_strict_dict_success(self):
        self.strict_dict.update({"key": "new_value"})
        self.assertEqual("new_value", self.strict_dict["key"])
