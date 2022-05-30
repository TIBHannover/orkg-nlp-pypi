from overrides import overrides

from orkgnlp.common.base import ORKGNLPBaseDecoder


class BioassaysSemantifierDecoder(ORKGNLPBaseDecoder):
    """
    TODO:
    """

    def __init__(self, mapping):
        super().__init__()

        self._mapping = mapping

    @overrides(check_signature=False)
    def decode(self, model_output, **kwargs):

        cluster_label = model_output[0][0]
        return self._mapping[str(cluster_label)]['labels']
