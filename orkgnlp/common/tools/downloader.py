"""
A script for downloading ORKG-NLP models and data needed to use the supported :doc:`../services/services`.

This script depends mainly on the ``huggingface_hub`` client and fetches the files from our
huggingface `repositories <https://huggingface.co/orkg>`_.
"""

import os
import logging
import shutil

from huggingface_hub import hf_hub_download

from orkgnlp.common.config import orkgnlp_context
from orkgnlp.common.util.exceptions import ORKGNLPValidationError

logger = logging.getLogger(__name__)


def _services_are_known(services, orkg_services):
    """
    Returns True if the given services names are subset of the known ORKG-NLP services

    :param services: list of service names
    :type services: list[str]
    :param orkg_services: list of ORKG-NLP service names
    :type services: list[str]
    :return:
    """
    return set(services).issubset(orkg_services)


def download(services):
    """
    Downloads the models and data needed to use the supported services.

    The download destination is given by ``ORKG_NLP_DATA_CACHE_ROOT``.
    You can also check how to :doc:`../configure` its value.

    :param services: a string representing a service name or a list of them. Check :doc:`../services/services` for a full list.
    :type services: str or list[str]
    :raise orkgnlp.common.util.exceptions.ORKGNLPValidationError: If one of the known passed service names is unknown.
    """
    if isinstance(services, str):
        services = [services]

    orkg_services = orkgnlp_context.get('HUGGINGFACE_REPOS')
    if not _services_are_known(services, orkg_services.keys()):
        raise ORKGNLPValidationError('Unknown model name(s) given {}. Please check the following known services: {}'
                              .format(services, list(orkg_services.keys())))

    cache_root = orkgnlp_context.get('ORKG_NLP_DATA_CACHE_ROOT')
    logger.info('Downloading to {}'.format(cache_root))
    for service in services:
        for repo in orkg_services[service]:
            for filename in repo['files'].values():
                hf_hub_download(
                    repo_id=repo['repo_id'],
                    filename=filename,
                    force_filename=filename,
                    cache_dir=os.path.join(cache_root, service)
                )


def exists_or_download(service):
    """
    Checks the presence of the required files for executing the given service and downloads them in case of absence.

    :param service: a string representing a ORKG-NLP service name. Check :doc:`../services/services` for a full list.
    :type service: str
    :return:
    """
    service_dir = os.path.join(orkgnlp_context.get('ORKG_NLP_DATA_CACHE_ROOT'), service)
    if not os.path.exists(service_dir):
        download(service)


def force_download(service):
    """
    Removes pre-downloaded files and downloads them again.

    :param service: a string representing a ORKG-NLP service name. Check :doc:`../services/services` for a full list.
    :type service: str
    :return:
    """
    service_dir = os.path.join(orkgnlp_context.get('ORKG_NLP_DATA_CACHE_ROOT'), service)

    if os.path.exists(service_dir):
        shutil.rmtree(service_dir)

    download(service)
