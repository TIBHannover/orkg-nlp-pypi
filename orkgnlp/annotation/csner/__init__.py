""" CS-NER package. """
import io

from orkgnlp.annotation.csner.annotator import CSNer

import spacy
import contextlib

with contextlib.redirect_stdout(io.StringIO()):
    spacy.cli.download('en_core_web_md', False, False, '--quiet')
