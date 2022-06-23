"""
Provides a set of global variables within the orkgnlp_context.
"""

import os

from orkgnlp.common.util import io
from orkgnlp.common.util.datastructure import StrictDict

_current_dir = os.path.dirname(os.path.realpath(__file__))

"""
Context dictionary for library's global configuration variables. 
Keys represent environment variables and values represent default values. 
Values can be changed using the corresponding setter function.
"""
orkgnlp_context = StrictDict(
    {
        'ORKG_NLP_DATA_CACHE_ROOT': os.path.join(os.path.expanduser('~'), 'orkgnlp_data'),
        'ORKG_NLP_VERBOSITY': True,
        'HUGGINGFACE_REPOS': io.read_json(os.path.join(_current_dir, '..', '..', 'huggingface_repos.json'))
    }
)


def set_data_cache_root(cache_root: str):
    """
    Overrides the value of ORKG_NLP_DATA_CACHE_ROOT. See :doc:`../configure`

    :param cache_root: Path to the data cache root. The path must be absolute.
    """
    orkgnlp_context['ORKG_NLP_DATA_CACHE_ROOT'] = cache_root


def set_verbosity(verbose: bool):
    """
    Overrides the value of ORKG_NLP_VERBOSITY. See :doc:`../configure`

    :param verbose: Indicates whether orkgnlp is in verbose mode.
    """
    orkgnlp_context['ORKG_NLP_VERBOSITY'] = verbose
