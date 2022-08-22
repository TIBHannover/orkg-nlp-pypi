""" Templates Recommendation service encoder. """

import torch

from typing import Dict, Any, Tuple, List
from overrides import overrides
from transformers import BertTokenizer

from orkgnlp.common.service.base import ORKGNLPBaseEncoder
from orkgnlp.common.util import text


class TemplatesRecommenderEncoder(ORKGNLPBaseEncoder):
    """
    The TemplatesRecommenderEncoder encodes the given input to the arguments
    needed to execute a BertForSequenceClassification model.
    """

    def __init__(self, templates: List[Dict[str, str]]):
        """

        :param templates: templates used for training the service models as premises.
        """
        super().__init__()

        self.templates: List[Dict[str, str]] = templates
        self.tokenizer: BertTokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
        self.max_input_sizes: int = self.tokenizer.max_model_input_sizes['bert-base-uncased'] or 512
        self.device: str = 'cpu'

    @overrides
    def encode(self, raw_input: Any, **kwargs: Any) -> Tuple[Any, Dict[str, Any]]:

        def batch_generator():
            for template in self.templates:
                sequence = '[CLS] {} [SEP] {} [SEP]'.format(self._post_process(template['premise']),
                                                            self._post_process(raw_input))
                sequence_tokens = self.tokenizer.tokenize(sequence)

                attention_mask = self._get_attention_mask(sequence_tokens)[:self.max_input_sizes]
                token_type = self._get_token_type(sequence_tokens)[:self.max_input_sizes:]
                sequence_tokens = self.tokenizer.convert_tokens_to_ids(sequence_tokens)[:self.max_input_sizes:]

                attention_mask = torch.tensor(attention_mask).unsqueeze(0).to(self.device)
                token_type = torch.tensor(token_type).unsqueeze(0).to(self.device)
                sequence_tokens = torch.tensor(sequence_tokens).unsqueeze(0).to(self.device)

                yield {
                    'input_ids': sequence_tokens,
                    'token_type_ids': token_type,
                    'attention_mask': attention_mask
                }

        return batch_generator(), kwargs

    @staticmethod
    def _get_attention_mask(tokens):
        return [1] * len(tokens)

    @staticmethod
    def _get_token_type(tokens):
        sep_index = tokens.index('[SEP]') + 1
        return [0] * sep_index + [1] * (len(tokens) - sep_index)

    @staticmethod
    def _post_process(string):
        string = text.replace(string, ['\s+-\s+', '-', '_', '\.'], ' ')
        return text.trim(string).lower()
