""" Base interfaces. """
from overrides import EnforceOverrides

from orkgnlp.common.service.executors import PipelineExecutor
from orkgnlp.common.tools import downloader
from orkgnlp.common.util.decorators import singleton
from orkgnlp.common.util.exceptions import ORKGNLPIllegalStateError, ORKGNLPValidationError


class ORKGNLPBaseService:
    """
        Base class for shared config parameters and functionalities. All ORKG-NLP services must inherit from this class.

        This class follows the singleton pattern,  i.e. only one instance can be obtained from it or its subclasses.
    """

    @singleton
    def __new__(cls):
        pass

    def __init__(self, service, *, force_download=False, batch_size=16):
        """

        :param force_download: Indicates whether the required files are to be downloaded again. Defaults to False.
        :type force_download: bool
        :param batch_size: Size of the batches used during model prediction. This argument is used by services
            that applies batch evaluation. Defaults to 16.
        :type batch_size: int
        """
        self._pipeline_executors = {}
        self._force_download = force_download
        self._batch_size = batch_size
        self._download(service)

    def _download(self, service):
        """
        Downloads the required files for the given service name based on the ``force_download`` class attribute.

        :param service: a string representing a ORKG-NLP service name. Check :doc:`../services/services` for a full list.
        :type service: str
        :return:
        """
        downloader.download(service, self._force_download)

    def _run(self, raw_input, pipline_executor_name=None, **kwargs):
        """
        Executes the only PipelineExecutor registered for the service, or one of them given its name.

        :param raw_input: User's input to be encoded.
        :type raw_input: Any.
        :param pipline_executor_name: Name of the PipelineExecutor to run.
        :type pipline_executor_name: str.
        :param kwargs: Named parameters for further processing config. Please check your used component documentation
            for specific parameter description.
        :type kwargs: Dict[str, Any].
        :return: The decoded user-friendly output.
        :raise orkgnlp.common.util.exceptions.ORKGNLPIllegalStateException: If either [Encoder, Runner, Decoder] is not
            initialized.
        :raise orkgnlp.common.util.exceptions.ORKGNLPValidationError: If the given ``pipline_executor_name`` is unknown
            or not given, in case of multiple registered ones.
        """
        if not self._pipeline_executors:
            raise ORKGNLPIllegalStateError('There is no PipelineRunner registered. Please consider registering'
                                           'one using the _register_pipline_runner() function.')

        if pipline_executor_name:

            if pipline_executor_name not in self._pipeline_executors:
                raise ORKGNLPValidationError('PipelineExecutor name is unknown.')

            return self._pipeline_executors[pipline_executor_name].run(raw_input, **kwargs)

        if len(self._pipeline_executors) > 1:
            raise ORKGNLPValidationError('PipelineExecutor is ambiguous. Consider passing pipline_executor_name in the '
                                         'input.')

        return next(iter(self._pipeline_executors.values())).run(raw_input, **kwargs)

    def _register_pipeline(self, name, encoder, runner, decoder):
        """
        Registers a PipelineExecutor to the service.

        :param name: PipelineExecutors name.
        :type name: str.
        :param encoder: Service's encoder.
        :type encoder: orkgnlp.common.service.base.ORKGNLPBaseEncoder.
        :param runner: Service's runner.
        :type runner: orkgnlp.common.service.base.ORKGNLPBaseRunner.
        :param decoder: Service's decoder.
        :type decoder: orkgnlp.common.service.base.ORKGNLPBaseDecoder.
        """
        if name in self._pipeline_executors:
            raise ORKGNLPValidationError('PipelineExecutor name already exists.')

        self._pipeline_executors[name] = PipelineExecutor(encoder, runner, decoder)


class ORKGNLPBaseEncoder(EnforceOverrides):
    """
    The ORKGNLPBaseEncoder is  the base encoder class. You can freely inherit this class, implement its
    ``encode(raw_input, **kwargs)`` function and use it to encode your user input to a model-friendly format.

    Using the ORKGNLPBaseEncoder as your service encoder results in passing the same user's input to the model.
    """

    def __init__(self):
        pass

    def encode(self, raw_input, **kwargs):
        """
        Encodes the ``raw_input`` to a model-friendly format.

        :param raw_input: The user's input to be encoded.
        :type raw_input: Any.
        :return: The model-friendly output and kwargs.
        """
        return raw_input, kwargs


class ORKGNLPBaseDecoder(EnforceOverrides):
    """
    The ORKGNLPBaseDecoder is  the base decoder class. You can freely inherit this class, implement its
    ``decode(model_output, **kwargs)`` function and use it to decode your model output to a user-friendly format.

    Using the ORKGNLPBaseDecoder as your service decoder results in returning the same model's output to the user.
    """

    def __init__(self):
        pass

    def decode(self, model_output, **kwargs):
        """
        Decodes the model's ``output`` to a user-friendly format.

        :param model_output: The model's output to be decoded.
        :type model_output: Any.
        :return: The user-friendly output.
        """
        return model_output, kwargs


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
