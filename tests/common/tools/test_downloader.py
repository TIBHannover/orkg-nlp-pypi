import os
from unittest import TestCase

from orkgnlp.common.config import orkgnlp_context
from orkgnlp.common.tools import downloader
from orkgnlp.common.util.exceptions import ORKGNLPValidationError


class TestDownloader(TestCase):

    def test_services_are_known(self):
        orkg_services = orkgnlp_context.get('HUGGINGFACE_REPOS')
        services = ['predicates-clustering']

        self.assertTrue(downloader._services_are_known(services, orkg_services))

        services = ['unknown']
        self.assertFalse(downloader._services_are_known(services, orkg_services))

    def test_delete_services(self):
        service_name = 'bioassays-semantification'
        service_dir = os.path.join(orkgnlp_context.get('ORKG_NLP_DATA_CACHE_ROOT'), service_name)
        services = [service_name]

        downloader.download(services)
        self.assertTrue(os.path.exists(service_dir))

        downloader._delete_services(services)
        self.assertFalse(os.path.exists(service_dir))

    def test_download(self):
        service = 'bioassays-semantification'
        root_dir = orkgnlp_context.get('ORKG_NLP_DATA_CACHE_ROOT')
        file_paths = []

        for file in orkgnlp_context.get('HUGGINGFACE_REPOS')[service][0]['files'].values():
            file_paths.append(
                os.path.join(root_dir, service, file)
            )

        downloader.download(service)

        for file in file_paths:
            self.assertTrue(os.path.exists(file))

        self.assertRaises(ORKGNLPValidationError, downloader.download, 'unknown')

