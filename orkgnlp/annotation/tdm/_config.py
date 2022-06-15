""" Config file for the tdm-extraction service. """

import os

from orkgnlp.common.config import orkgnlp_context
from orkgnlp.common.util.datastructure import StrictDict

_service_data_dir = os.path.join(
    orkgnlp_context.get('ORKG_NLP_DATA_CACHE_ROOT'),
    orkgnlp_context.get('TDM_EXTRACTION_SERVICE_NAME')
)

_service_data_files = {}
for repo in orkgnlp_context.get('HUGGINGFACE_REPOS')[orkgnlp_context.get('TDM_EXTRACTION_SERVICE_NAME')]:
    _service_data_files.update(repo['files'])

config = StrictDict({
    'service_name': orkgnlp_context.get('TDM_EXTRACTION_SERVICE_NAME'),
    'paths': {
        'model_dir': _service_data_dir,
        'labels': os.path.join(_service_data_dir, _service_data_files['labels'])
    }
})
