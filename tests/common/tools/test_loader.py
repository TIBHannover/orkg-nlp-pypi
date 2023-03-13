# -*- coding: utf-8 -*-
from unittest import TestCase

import orkgnlp
from orkgnlp.common.config import orkgnlp_context
from orkgnlp.common.util.exceptions import ORKGNLPUnknownServiceError


class TestLoader(TestCase):
    def test_load_with_unknown_name(self):
        self.assertRaises(ORKGNLPUnknownServiceError, orkgnlp.load, "unknown_service_name")

    def test_load(self):
        for service_name, service_cls in orkgnlp_context["SERVICE_MAP"].items():
            service_object = orkgnlp.load(service_name)
            self.assertIsInstance(service_object, service_cls)
            service_object.release_memory()
