""" Base interfaces. """
import os
from typing import Any, Dict, Tuple, Union, Generator

from overrides import EnforceOverrides

from orkgnlp.common.config import orkgnlp_context
from orkgnlp.common.tools import downloader
from orkgnlp.common.util.decorators import singleton
from orkgnlp.common.util.exceptions import ORKGNLPIllegalStateError, ORKGNLPValidationError


class PipelineExecutorComponent:
    """
    The PipelineExecutorComponent represents a component of a PipelineExecutor
    """
    def release_memory(self):
        """
        Releases the memory of all available attributes in a pipeline component.
        """
        attributes = list(self.__dict__.keys())
        for attribute in attributes:
            delattr(self, attribute)


class ORKGNLPBaseEncoder(EnforceOverrides, PipelineExecutorComponent):
    """
    The ORKGNLPBaseEncoder is  the base encoder class. You can freely inherit this class, implement its
    ``encode(raw_input, **kwargs)`` function and use it to encode your user input to a model-friendly format.

    Using the ORKGNLPBaseEncoder as your service encoder results in passing the same user's input to the model.
    """

    def __init__(self):
        pass

    def encode(self, raw_input: Any, **kwargs: Any) -> Tuple[Any, Dict[str, Any]]:
        """
        Encodes the ``raw_input`` to a model-friendly format.

        :param raw_input: The user's input to be encoded.
        :return: The model-friendly output and kwargs.
        """
        return (raw_input, ), kwargs


class ORKGNLPBaseDecoder(EnforceOverrides, PipelineExecutorComponent):
    """
    The ORKGNLPBaseDecoder is  the base decoder class. You can freely inherit this class, implement its
    ``decode(model_output, **kwargs)`` function and use it to decode your model output to a user-friendly format.

    Using the ORKGNLPBaseDecoder as your service decoder results in returning the same model's output to the user.
    """

    def __init__(self):
        pass

    def decode(
            self,
            model_output: Union[Any, Generator[Any, None, None]],
            **kwargs: Any
    ) -> Any:
        """
        Decodes the model's ``output`` to a user-friendly format.

        :param model_output: The model's output to be decoded.
        :return: The user-friendly output.
        """
        return model_output, kwargs


class ORKGNLPBaseRunner(EnforceOverrides, PipelineExecutorComponent):
    """
    The ORKGNLPBaseRunner is  the base runner class. It requires a model object while initialization.
    This runner must be inherited and the `run(*args, **kwargs)` must be overridden,
    thus running this runner raises an ``NotImplementedError``.
    """

    def __init__(self, model: Any):
        """

        :param model: The model to be run.
        :type model: Model object. See the inheriting classes for further information.
        """
        self._model: Any = model

    def run(self, *args: Any, **kwargs: Any):
        """

        :raise: NotImplementedError
        """
        raise NotImplementedError('Subclass must implement abstract method')


class PipelineExecutor:
    """
    The PipelineExecutor executes a full service workflow given its encoder, runner and decoder.
    See the ``run`` function description for further information.
    """
    def __init__(self, encoder: ORKGNLPBaseEncoder, runner: ORKGNLPBaseRunner, decoder: ORKGNLPBaseDecoder):
        """

        :param encoder: Service's encoder.
        :param runner: Service's runner.
        :param decoder: Service's decoder.
        """
        self._encoder: ORKGNLPBaseEncoder = encoder
        self._runner: ORKGNLPBaseRunner = runner
        self._decoder: ORKGNLPBaseDecoder = decoder

    def run(self, raw_input: Any, **kwargs: Any) -> Any:
        """
        Executes a full pipline of the common service workflow:

        1. Runs the service encoder with the user's input.
        2. The encoded input is passed to the model runner, which in turn is executed.
        3. The model's output is decoded to a user-friendly format using the service's decoder.

        Note that the kwargs can be updated and passed through the pipeline components.

        :param raw_input: User's input to be encoded.
        :param kwargs: Named parameters for further processing config. Please check your used component documentation
            for specific parameter description.
        :return: The decoded user-friendly output.
        :raise orkgnlp.common.util.exceptions.ORKGNLPIllegalStateException: If either [Encoder, Runner, Decoder] is not
            initialized.
        """
        if not (self._encoder and self._runner and self._decoder):
            raise ORKGNLPIllegalStateError('Encoder, Runner and Decoder must be initialized!')

        inputs, additional_properties = self._encoder.encode(raw_input, **kwargs)
        kwargs.update(additional_properties)

        output, additional_properties = self._runner.run(inputs, **kwargs)
        kwargs.update(additional_properties)

        return self._decoder.decode(output, **kwargs)

    def release_memory(self):
        """
        Releases the memory of all available pipeline components.
        """
        self._encoder.release_memory()
        self._runner.release_memory()
        self._decoder.release_memory()


class ORKGNLPBaseConfig:
    """
    The ORKGNLPBaseConfig encapsulates the required configurations for a service given its name.
    """

    def __init__(self, service: str):
        """

        :param service: The service name.
        """
        service_data_dir = os.path.join(
            orkgnlp_context.get('ORKG_NLP_DATA_CACHE_ROOT'),
            service
        )

        self.service_name: str = service
        self.service_dir: str = service_data_dir
        self.requirements = self._get_requirement_paths()

    def _get_requirement_paths(self) -> Dict[str, str]:
        paths = {}
        for repo in orkgnlp_context.get('HUGGINGFACE_REPOS')[self.service_name]:
            for file_obj in repo['files']:
                paths[file_obj['internal_name']] = os.path.join(
                    self.service_dir, file_obj.get('subbdir', ''), file_obj['filename']
                )

        return paths


class ORKGNLPBaseService:
    """
        Base class for shared config parameters and functionalities. All ORKG-NLP services must inherit from this class.

        This class follows the singleton pattern,  i.e. only one instance can be obtained from it or its subclasses.
    """

    @singleton
    def __new__(cls, *args, **kwargs):
        pass

    def __init__(self, service: str, *, force_download: bool = False, batch_size: int = 16, _unittest: bool = False):
        """

        :param service: Service name.
        :param force_download: Indicates whether the required files are to be downloaded again. Defaults to False.
        :param batch_size: Size of the batches used during model prediction. This argument is used by services
            that applies batch evaluation. Defaults to 16.
        """
        self._pipeline_executors: Dict[str, PipelineExecutor] = {}
        self._config: ORKGNLPBaseConfig = ORKGNLPBaseConfig(service)
        self._force_download: bool = force_download
        self._batch_size: int = batch_size
        self._unittest: bool = _unittest
        self._download(service)

    def _download(self, service: str):
        """
        Downloads the required files for the given service name based on the ``force_download`` class attribute.

        :param service: a string representing a ORKG-NLP service name. Check :doc:`../services/services` for a full list.
        :return:
        """
        downloader.download(service, self._force_download)

    def _run(self, raw_input: Any, pipline_executor_name: str = None, **kwargs: Any):
        """
        Executes the only PipelineExecutor registered for the service, or one of them given its name.

        :param raw_input: User's input to be encoded.
        :param pipline_executor_name: Name of the PipelineExecutor to run.
        :param kwargs: Named parameters for further processing config. Please check your used component documentation
            for specific parameter description.
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

    def _register_pipeline(
            self,
            name: str,
            encoder: ORKGNLPBaseEncoder,
            runner: ORKGNLPBaseRunner,
            decoder: ORKGNLPBaseDecoder
    ):
        """
        Registers a PipelineExecutor to the service.

        :param name: PipelineExecutors name.
        :param encoder: Service's encoder.
        :param runner: Service's runner.
        :param decoder: Service's decoder.
        """
        if name in self._pipeline_executors:
            raise ORKGNLPValidationError('PipelineExecutor name already exists.')

        self._pipeline_executors[name] = PipelineExecutor(encoder, runner, decoder)

    def release_memory(self):
        """
        Releases the memory of all available executors.
        """
        for executor in self._pipeline_executors.values():
            executor.release_memory()
