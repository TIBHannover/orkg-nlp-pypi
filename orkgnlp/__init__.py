# -*- coding: utf-8 -*-
"""
Root package of orkgnlp.
"""

__version__ = "0.11.1"

import logging

from orkgnlp import annotation, clustering, nli
from orkgnlp.common import config, service, tools, util
from orkgnlp.common.tools import download, load

__all__ = [
    "annotation",
    "clustering",
    "nli",
    "config",
    "service",
    "tools",
    "util",
    "download",
    "load",
]


# Root logger configuration
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

stdout = logging.StreamHandler()
stdout.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(name)s - %(levelname)s: %(message)s")
stdout.setFormatter(formatter)

logger.addHandler(stdout)
