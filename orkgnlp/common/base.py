""" Base interfaces. """

from orkgnlp.common.tools import downloader


class ORKGNLPBase:
    """
        Base class for shared config parameters.
    """
    def __init__(self, service, force_download=False):
        """

        :param force_download: Indicates whether the required files are to be downloaded again. Defaults to False.
        :type force_download: bool
        """
        self.force_download = force_download
        self._download(service)

    def _download(self, service):
        """
        Downloads the required files for the given service name based on the ``force_download`` class attribute.

        :param service: a string representing a ORKG-NLP service name. Check :doc:`../services/services` for a full list.
        :type service: str
        :return:
        """
        if self.force_download:
            downloader.force_download(service)
        else:
            downloader.exists_or_download(service)
