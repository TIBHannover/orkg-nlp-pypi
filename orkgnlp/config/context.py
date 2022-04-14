"""
Provides a set of global variables within the orkgnlp_context
"""

import os

from orkgnlp.util.datastructure import StrictDict

"""
Context dictionary for library's global configuration variables. 
Keys represent environment variables and values represent default values. 
Values can be changed using the corresponding setter function.
"""
orkgnlp_context = StrictDict(
    {
        'ORKG_NLP_DATA_CACHE_ROOT': os.path.join(os.path.expanduser('~'), 'orkgnlp_data')
    }
)


def set_data_cache_root(cache_root):
    """
    Overrides the value of ORKG_NLP_DATA_CACHE_ROOT. See :doc:`../configure`

    :param cache_root: Path to the data cache root. The path must be absolute.
    :type cache_root: str
    """
    orkgnlp_context['ORKG_NLP_DATA_CACHE_ROOT'] = cache_root
