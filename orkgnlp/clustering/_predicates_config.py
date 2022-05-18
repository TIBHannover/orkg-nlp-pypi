""" Config file for the predicates clustering service. """

import os

from orkgnlp.common.config import orkgnlp_context
from orkgnlp.common.util.datastructure import StrictDict

_service_data_dir = os.path.join(
    orkgnlp_context.get('ORKG_NLP_DATA_CACHE_ROOT'),
    orkgnlp_context.get('PREDICATES_CLUSTERING_SERVICE_NAME')
)
_service_data_files = \
    orkgnlp_context.get('HUGGINGFACE_REPOS')[orkgnlp_context.get('PREDICATES_CLUSTERING_SERVICE_NAME')][0]['files']

config = StrictDict({
    'service_name': orkgnlp_context.get('PREDICATES_CLUSTERING_SERVICE_NAME'),
    'paths': {
        'vectorizer': os.path.join(_service_data_dir, _service_data_files['vectorizer']),
        'model': os.path.join(_service_data_dir, _service_data_files['model']),
        'training_data': os.path.join(_service_data_dir, _service_data_files['training_data']),
        'mapping': os.path.join(_service_data_dir, _service_data_files['mapping'])
    }
})
