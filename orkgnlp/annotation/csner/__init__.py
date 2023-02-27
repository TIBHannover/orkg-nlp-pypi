# -*- coding: utf-8 -*-
""" CS-NER package. """
import contextlib
import io

import spacy

from orkgnlp.annotation.csner.annotator import CSNer

with contextlib.redirect_stdout(io.StringIO()):
    spacy.cli.download("en_core_web_md", False, False, "--quiet")

__all__ = ["CSNer"]
