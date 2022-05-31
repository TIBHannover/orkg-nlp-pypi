""" Model runners. """

import onnxruntime as rt
from overrides import overrides

from orkgnlp.common.service.base import ORKGNLPBaseRunner


class ORKGNLPONNXRunner(ORKGNLPBaseRunner):
    """
    The ORKGNLPONNXRunner is a runner specialized for ONNX model formats. It requires therefore a model object of type
    ``onnx``.
    """
    def __init__(self, *args):
        super().__init__(*args)

    @overrides(check_signature=False)
    def run(self, inputs, output_names=None, custom_input_dict=None, **kwargs):
        """
        Runs the given model while initiation in evaluation mode and returns its output.

        :param inputs: Tuple of model arguments.
        :type inputs: Tuple[Any].
        :param output_names: List of output names of the ONNX graph. Check your exporting code for further information!
            Defaults to None.
        :type output_names: List[str].
        :param custom_input_dict: When given, the argument ``inputs`` will be ignored. This argument must have the
            following schema: {input_name_0: [input_value_0], ..., input_name_n: [input_value_n]}. Check your exporting
            code for further information! Defaults to None.
        :type custom_input_dict: Dict[str, List[Any]].
        :return: The model output and kwargs.
        """

        session = rt.InferenceSession(self._model.SerializeToString())
        input_dict = {session.get_inputs()[i].name: [inputs[i]] for i in range(len(inputs))}
        output = session.run(output_names, custom_input_dict or input_dict)

        return output, kwargs


class ORKGNLPTorchRunner(ORKGNLPBaseRunner):
    """
    The ORKGNLPTorchRunner is a runner specialized for Torch model formats. It requires therefore a model object of type
    ``torch``.
    """
    def __init__(self, *args):
        super().__init__(*args)

    @overrides(check_signature=False)
    def run(self, inputs, batch_eval=False, **kwargs):
        """
        Runs the given model while initiation in evaluation mode and returns its output.

        :param inputs: Tuple of model arguments or a list of tuples if ``batch_eval=True``.
        :type inputs: Tuple[Any] or List[Tuple[Any]].
        :param batch_eval: Whether the model is to be executed x times for x batches, where
            x is the length of ``inputs`` list. Defaults to False.
        :type batch_eval: bool
        :return: The model output as a tuple or list of tuples, and kwargs.
        """
        self._model.eval()

        if not batch_eval:
            return self._model(*inputs)

        outputs = []
        for batch in inputs:
            outputs.append([*self._model(*batch)])

        return outputs, kwargs
