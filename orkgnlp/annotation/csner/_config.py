""" Config file for the cs-ner service. """

import os

from orkgnlp.common.config import orkgnlp_context
from orkgnlp.common.util.datastructure import StrictDict

_service_data_dir = os.path.join(
    orkgnlp_context.get('ORKG_NLP_DATA_CACHE_ROOT'),
    orkgnlp_context.get('CS_NER_SERVICE_NAME')
)
_service_data_files = {}
for repo in orkgnlp_context.get('HUGGINGFACE_REPOS')[orkgnlp_context.get('CS_NER_SERVICE_NAME')]:
    _service_data_files.update(repo['files'])

config = StrictDict({
    'service_name': orkgnlp_context.get('CS_NER_SERVICE_NAME'),
    'paths': {
        'titles_model': os.path.join(_service_data_dir, _service_data_files['titles_model']),
        'titles_alphabet': os.path.join(_service_data_dir, _service_data_files['titles_alphabet']),
        'abstracts_model': os.path.join(_service_data_dir, _service_data_files['abstracts_model']),
        'abstracts_alphabet': os.path.join(_service_data_dir, _service_data_files['abstracts_alphabet'])
    }
})
