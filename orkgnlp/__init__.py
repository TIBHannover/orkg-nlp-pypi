"""
Root package of orkgnlp.
"""

__version__ = '0.5.0'

import logging

from orkgnlp.common import *
from orkgnlp.common.tools import *

# Root logger configuration
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

stdout = logging.StreamHandler()
stdout.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(name)s - %(levelname)s: %(message)s')
stdout.setFormatter(formatter)

logger.addHandler(stdout)
