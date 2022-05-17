""" Config file for the bioassays semantification service. """

import os

from orkgnlp.config import orkgnlp_context
from orkgnlp.util.datastructure import StrictDict

_service_data_dir = os.path.join(
    orkgnlp_context.get('ORKG_NLP_DATA_CACHE_ROOT'),
    orkgnlp_context.get('BIOASSAYS_SEMANTIFICATION_SERVICE_NAME')
)
_service_data_files = \
    orkgnlp_context.get('HUGGINGFACE_REPOS')[orkgnlp_context.get('BIOASSAYS_SEMANTIFICATION_SERVICE_NAME')][0]['files']

config = StrictDict({
    'service_name': orkgnlp_context.get('BIOASSAYS_SEMANTIFICATION_SERVICE_NAME'),
    'paths': {
        'vectorizer': os.path.join(_service_data_dir, _service_data_files['vectorizer']),
        'model': os.path.join(_service_data_dir, _service_data_files['model']),
        'mapping': os.path.join(_service_data_dir, _service_data_files['mapping'])
    }
})
