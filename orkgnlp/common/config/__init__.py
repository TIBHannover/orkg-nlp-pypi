# -*- coding: utf-8 -*-
"""
Config package that allows a global configuring of orkg-nlp.
"""

from orkgnlp.common.config.context import (
    orkgnlp_context,
    set_data_cache_root,
    set_verbosity,
)

__all__ = ["orkgnlp_context", "set_verbosity", "set_data_cache_root"]
