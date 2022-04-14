"""
A script for downloading ORKG-NLP models and data needed to use the supported :doc:`../services`.

This script depends mainly on the ``huggingface_hub`` client and fetches the files from our
huggingface `repositories <https://huggingface.co/orkg>`_.
"""

import os
import logging
from huggingface_hub import hf_hub_download

from orkgnlp.config import orkgnlp_context
from orkgnlp.util import io
from orkgnlp.util.exceptions import ORKGNLPValidationError

logger = logging.getLogger(__name__)
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


def _repos_are_known(services, orkg_services):
    return set(services).issubset(orkg_services.keys())


def download(services):
    """
    Downloads the models and data needed to use the supported services.

    The download destination is given by ``ORKG_NLP_DATA_CACHE_ROOT``.
    You can also check how to :doc:`../configure` its value.

    :param services: a string representing a service name or a list of them. Check :doc:`../services` for a full list.
    :type services: str or list[str]
    :raise orkgnlp.util.exceptions.ORKGNLPValidationError: If one of the known passed service names is unknown.
    """
    if isinstance(services, str):
        services = [services]

    orkg_services = io.read_json(os.path.join(CURRENT_DIR, '../huggingface_repos.json'))
    if not _repos_are_known(services, orkg_services):
        raise ORKGNLPValidationError('Unknown model name(s) given {}. Please check the following known services: {}'
                              .format(services, list(orkg_services.keys())))

    cache_root = orkgnlp_context.get('ORKG_NLP_DATA_CACHE_ROOT')
    logger.info('Downloading to {}'.format(cache_root))
    for model in services:
        for repo in orkg_services[model]:
            for filename in repo['files']:
                hf_hub_download(
                    repo_id=repo['repo_id'],
                    filename=filename,
                    force_filename=filename,
                    cache_dir=os.path.join(cache_root, model)
                )
