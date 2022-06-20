""" BioAssays semantification service decoder. """

from overrides import overrides
from typing import Any, Dict, Union, Tuple, Generator

from orkgnlp.common.service.base import ORKGNLPBaseDecoder


class BioassaysSemantifierDecoder(ORKGNLPBaseDecoder):
    """
    The BioassaysSemantifierDecoder decodes the Bioassays semantification service model's output
    to a user-friendly one.
    """
    def __init__(self, mapping: Dict[str, Dict[str, Any]]):
        """

        :param mapping: Cluster label resources mapping.
        :type mapping: Dict[str, Dict[str, Any]]
        """
        super().__init__()

        self._mapping = mapping

    @overrides
    def decode(
            self,
            model_output: Union[Any, Generator[Any, None, None]],
            **kwargs: Any
    ) -> Any:

        cluster_label = model_output[0][0]
        return self._mapping[str(cluster_label)]['labels']
