import os
import logging
from huggingface_hub import hf_hub_download

from orkgnlp.config import orkgnlp_context
from orkgnlp.util import io
from orkgnlp.util.exceptions import ValidationError

logger = logging.getLogger(__name__)
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


def _repos_are_known(model_names, orkg_models):
    return set(model_names).issubset(orkg_models.keys())


def download(model_names):
    if isinstance(model_names, str):
        model_names = [model_names]

    orkg_models = io.read_json(os.path.join(CURRENT_DIR, '../huggingface_repos.json'))
    if not _repos_are_known(model_names, orkg_models):
        raise ValidationError('Unknown model name(s) given {}. Please check the following known models: {}'
                              .format(model_names, list(orkg_models.keys())))

    cache_root = orkgnlp_context.get('ORKG_NLP_DATA_CACHE_ROOT')
    logger.info('Downloading to {}'.format(cache_root))
    for model_name in model_names:
        for repo in orkg_models[model_name]:
            for filename in repo['files']:
                hf_hub_download(
                    repo_id=repo['repo_id'],
                    filename=filename,
                    force_filename=filename,
                    cache_dir=os.path.join(cache_root, model_name)
                )

