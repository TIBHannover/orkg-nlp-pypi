# -*- coding: utf-8 -*-
"""
Root package of orkgnlp.
"""

__version__ = "0.9.0"

import logging

from orkgnlp.common import config, service, tools, util
from orkgnlp.common.tools import download

__all__ = ["config", "service", "tools", "util", "download"]


# Root logger configuration
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

stdout = logging.StreamHandler()
stdout.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(name)s - %(levelname)s: %(message)s")
stdout.setFormatter(formatter)

logger.addHandler(stdout)
