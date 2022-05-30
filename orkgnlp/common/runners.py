""" Model runners. """

import onnxruntime as rt
from overrides import overrides, EnforceOverrides

from orkgnlp.common.base import ORKGNLPBaseRunner


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
        :return: The model output.
        """

        session = rt.InferenceSession(self._model.SerializeToString())
        input_dict = {session.get_inputs()[i].name: [inputs[i]] for i in range(len(inputs))}
        output = session.run(output_names, custom_input_dict or input_dict)
        return output
