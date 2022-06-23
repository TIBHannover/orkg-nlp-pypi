"""
A script for downloading ORKG-NLP models and data needed to use the supported :doc:`../services/services`.

This script depends mainly on the ``huggingface_hub`` client and fetches the files from our
huggingface `repositories <https://huggingface.co/orkg>`_.
"""

import os
import shutil
from typing import Union, List

from huggingface_hub import hf_hub_download
from wasabi import msg

from orkgnlp.common.config import orkgnlp_context
from orkgnlp.common.util.exceptions import ORKGNLPValidationError


def download(services: Union[str, List[str]], force_download: bool = False):
    """
    Downloads the models and data needed to use the supported services.

    The download destination is given by ``ORKG_NLP_DATA_CACHE_ROOT``.
    You can also check how to :doc:`../configure` its value.

    :param services: a string representing a service name or a list of them. Check :doc:`../services/services` for a full list.
    :param force_download: Indicates whether the required files are to be downloaded again. Defaults to False.
    :raise orkgnlp.common.util.exceptions.ORKGNLPValidationError: If one of the known passed service names is unknown.
    """
    if isinstance(services, str):
        services = [services]

    if force_download:
        _delete_services(services)

    _download(services)


def _download(services: List[str]):
    """
    Checks whether the service names are known and then for each service it downloads the required
    not-already-downloaded files.

    :param services: A list representing service names. Check :doc:`../services/services` for a full list.
    :raise orkgnlp.common.util.exceptions.ORKGNLPValidationError: If one of the known passed service names is unknown.
    """

    orkg_services = orkgnlp_context.get('HUGGINGFACE_REPOS')
    if not _services_are_known(services, orkg_services.keys()):
        raise ORKGNLPValidationError('Unknown model name(s) given {}. Please check the following known services: {}'
                                     .format(services, list(orkg_services.keys())))

    cache_root = orkgnlp_context.get('ORKG_NLP_DATA_CACHE_ROOT')
    for service in services:
        already_found = True
        service_dir = os.path.join(cache_root, service)

        for repo in orkg_services[service]:
            for file_obj in repo['files']:

                if file_obj.get('subdir'):
                    filename = os.path.join(file_obj['subdir'], file_obj['filename'])
                    os.makedirs(os.path.join(service_dir, file_obj['subdir']), exist_ok=True)
                else:
                    filename = file_obj['filename']

                if not os.path.exists(os.path.join(service_dir, filename)):
                    hf_hub_download(
                        repo_id=repo['repo_id'],
                        filename=filename,
                        force_filename=filename,
                        cache_dir=service_dir
                    )
                    already_found = False

        if already_found:
            msg.good('Required files for "{}" service already found in {}. '
                     'Please consider passing the argument force_download=True in case you '
                     'need a fresh copy.'.format(service, service_dir),
                     show=orkgnlp_context.get('ORKG_NLP_VERBOSITY'))
        else:
            msg.good(service_dir,
                     show=orkgnlp_context.get('ORKG_NLP_VERBOSITY'))


def _services_are_known(services: List[str], orkg_services: List[str]) -> bool:
    """
    Returns True if the given services names are subset of the known ORKG-NLP services

    :param services: list of service names
    :param orkg_services: list of ORKG-NLP service names
    :return:
    """
    return set(services).issubset(orkg_services)


def _delete_services(services: List[str]):
    """
    Removes pre-downloaded files of the given service names.

    :param services: A list representing service names. Check :doc:`../services/services` for a full list.
    """

    for service in services:
        service_dir = os.path.join(orkgnlp_context.get('ORKG_NLP_DATA_CACHE_ROOT'), service)

        if os.path.exists(service_dir):
            shutil.rmtree(service_dir)
