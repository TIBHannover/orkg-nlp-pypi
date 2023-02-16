""" Agri-NER service decoder. """

from typing import Any, Dict, List

from overrides import overrides

from orkgnlp.common.service.base import ORKGNLPBaseDecoder


class AgriNerDecoder(ORKGNLPBaseDecoder):
    """
    The AgriNerDecoder decodes the Agri-NER service model's output
    to a user-friendly one.
    """

    CONCEPTS_MAP = {
        'RP': 'RESEARCH_PROBLEM',
        'P': 'PROCESS',
        'METH': 'METHOD',
        'R': 'RESOURCE',
        'S': 'SOLUTION',
        'LOC': 'LOCATION',
        'T': 'TECHNOLOGY'
    }

    @overrides(check_signature=False)
    def decode(
            self,
            model_output: List[Dict[str, Any]],
            **kwargs: Any
    ) -> Any:

        annotations = []
        seen = []
        for entity in model_output:

            if entity['entity_group'] in seen:
                continue

            seen.append(entity['entity_group'])
            annotations.append({
                'concept': self.CONCEPTS_MAP[entity['entity_group']],
                'entities': [i['word'] for i in model_output if i['entity_group'] == entity['entity_group']]
            })

        return annotations
