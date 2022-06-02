""" Service Executors. """

from orkgnlp.common.util.exceptions import ORKGNLPIllegalStateError


class PipelineExecutor:
    """
    The PipelineExecutor executes a full service workflow given its encoder, runner and decoder.
    See the ``run`` function description for further information.
    """
    def __init__(self, encoder, runner, decoder):
        """

        :param encoder: Service's encoder.
        :type encoder: orkgnlp.common.service.base.ORKGNLPBaseEncoder.
        :param runner: Service's runner.
        :type runner: orkgnlp.common.service.base.ORKGNLPBaseRunner.
        :param decoder: Service's decoder.
        :type decoder: orkgnlp.common.service.base.ORKGNLPBaseDecoder.
        """
        self._encoder = encoder
        self._runner = runner
        self._decoder = decoder

    def run(self, raw_input, **kwargs):
        """
        Executes a full pipline of the common service workflow:

        1. Runs the service encoder with the user's input.
        2. The encoded input is passed to the model runner, which in turn is executed.
        3. The model's output is decoded to a user-friendly format using the service's decoder.

        Note that the kwargs can be updated and passed through the pipeline components.

        :param raw_input: User's input to be encoded.
        :type raw_input: Any.
        :param kwargs: Named parameters for further processing config. Please check your used component documentation
            for specific parameter description.
        :type kwargs: Dict[str, Any].
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
