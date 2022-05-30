""" Base interfaces. """
from overrides import EnforceOverrides

from orkgnlp.common.tools import downloader
from orkgnlp.common.util.exceptions import ORKGNLPIllegalStateException


class ORKGNLPBaseService:
    """
        Base class for shared config parameters and functionalities.
    """
    def __init__(self, service, force_download=False):
        """

        :param force_download: Indicates whether the required files are to be downloaded again. Defaults to False.
        :type force_download: bool
        """

        self._encoder = None
        self._runner = None
        self._decoder = None
        self._force_download = force_download
        self._download(service)

    def _download(self, service):
        """
        Downloads the required files for the given service name based on the ``force_download`` class attribute.

        :param service: a string representing a ORKG-NLP service name. Check :doc:`../services/services` for a full list.
        :type service: str
        :return:
        """
        if self._force_download:
            downloader.force_download(service)
        else:
            downloader.exists_or_download(service)

    def _run(self, raw_input, **kwargs):
        """
        Executes a full pipline of the common service workflow:

        1. Runs the service encoder with the user's input.
        2. The encoded input is passed to the model runner, which in turn is executed.
        3. The model's output is decoded to a user-friendly format using the service's decoder.

        :param raw_input: User's input to be encoded.
        :type raw_input: Any.
        :param kwargs: Named parameters for further processing config. Please check your used component documentation
            for specific parameter description.
        :type kwargs: Dict[str, Any].
        :return: The decoded user-friendly output.
        :raise orkgnlp.common.util.exceptions.ORKGNLPIllegalStateException: if either [Encoder, Runner, Decoder] is not
            initialized.
        """
        if not (self._encoder and self._runner and self._decoder):
            raise ORKGNLPIllegalStateException('Encoder, Runner and Decoder must be initialized!')

        inputs = self._encoder.encode(raw_input, **kwargs)
        output = self._runner.run(inputs, **kwargs)
        return self._decoder.decode(output, **kwargs)


class ORKGNLPBaseEncoder(EnforceOverrides):
    """
    The ORKGNLPBaseEncoder is  the base encoder class. You can freely inherit this class, implement its
    ``encode(*args, **kwargs)`` function and use it to encode your user input to a model-friendly format.

    Using the ORKGNLPBaseEncoder as your service encoder results in passing the same user's input to the model.
    """

    def __init__(self):
        pass

    def encode(self, raw_input, **kwargs):
        """
        Encodes the ``raw_input`` to a model-friendly format.

        :param raw_input: The user's input to be encoded.
        :type raw_input: Any.
        :return: The model-friendly output.
        """
        return raw_input, kwargs


class ORKGNLPBaseDecoder(EnforceOverrides):
    """
    The ORKGNLPBaseDecoder is  the base decoder class. You can freely inherit this class, implement its
    ``decode(*args, **kwargs)`` function and use it to decode your model output to a user-friendly format.

    Using the ORKGNLPBaseDecoder as your service decoder results in returning the same model's output to the user.
    """

    def __init__(self):
        pass

    def decode(self, output, **kwargs):
        """
        Decodes the model ``output`` to a user-friendly format.

        :param output: The model's output to be decoded.
        :type output: Any.
        :return: The user-friendly output.
        """
        return output, kwargs


class ORKGNLPBaseRunner(EnforceOverrides):
    """
    The ORKGNLPBaseRunner is  the base runner class. It requires a model object while initialization.
    This runner must be inherited and the `run(*args, **kwargs)` must be overridden,
    thus running this runner raises an ``NotImplementedError``.
    """
    def __init__(self, model):
        """

        :param model: The model to be run.
        :type model: Model object. See the inheriting classes for further information.
        """
        self._model = model

    def run(self, *args, **kwargs):
        """

        :raise: NotImplementedError
        """
        raise NotImplementedError('Subclass must implement abstract method')
