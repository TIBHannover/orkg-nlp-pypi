""" TDM-Extraction service encoder. """

import torch
from overrides import overrides
from pandas import DataFrame
from torch.utils.data import Dataset, DataLoader
from transformers import XLNetTokenizer, PreTrainedTokenizer
from typing import Any, Dict, Tuple, List

from orkgnlp.common.service.base import ORKGNLPBaseEncoder


class TdmExtractorEncoder(ORKGNLPBaseEncoder):
    """
    The TdmExtractorEncoder encodes the given input to the arguments
    needed to execute an XLNetForSequenceClassification model.
    """

    def __init__(self, labels: DataFrame, batch_size: int):
        """

        :param labels: TDM gold labels given as one-columned-dataframe
        :param batch_size: Size of the batches used during model prediction.
        """
        super().__init__()

        self.labels: DataFrame = labels
        self.batch_size: int = batch_size
        self.tokenizer: XLNetTokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
        self.max_input_sizes: int = self.tokenizer.max_model_input_sizes['xlnet-base-cased'] or 512
        self.device: str = 'cpu'

    @overrides
    def encode(self, raw_input: Any, **kwargs: Any) -> Tuple[Any, Dict[str, Any]]:
        dataset = TdmDataset(raw_input, self.labels, self.tokenizer, self.max_input_sizes)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self._collate_batch)

        def batch_generator():
            for batch in dataloader:
                yield {
                    'input_ids': batch[0].to(self.device),
                    'token_type_ids': batch[1].to(self.device),
                    'attention_mask': batch[2].to(self.device)
                }

        return batch_generator(), kwargs

    @staticmethod
    def _collate_batch(batch: List[Tuple[Any, ...]]) -> Tuple[Any, ...]:
        max_shape = max([instance[0].shape[0] for instance in batch])

        args_num = len(batch[0])
        args = [[] for _ in range(args_num)]
        for instance in batch:
            for i in range(args_num):
                zero_padding = torch.zeros(max_shape - instance[i].shape[0], dtype=torch.int)
                args[i].append(torch.cat((instance[i], zero_padding)))

        return tuple(torch.stack(args[i]) for i in range(args_num))


class TdmDataset(Dataset):
    """
    The TdmDataset is a customized torch.utils.data.Dataset that simplifies the tokenization of sequences and
    can be used afterwards in a torch.utils.data.Dataloader for batch creation.
    """

    def __init__(self, text: str, labels: DataFrame, tokenizer: PreTrainedTokenizer, max_input_sizes: int):
        """

        :param text: Input text (hypothesis) to be concatenated with all known labels (premises).
        :param labels: TDM gold labels given as one-columned-dataframe
        :param tokenizer: Tokenizer for tokenizing the texts.
        :param max_input_sizes: Max length of a sequence including special characters.
        """
        self.labels: DataFrame = labels
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.hypothesis_ids: List[int] = self.tokenizer.encode(text, add_special_tokens=False)
        self.max_input_sizes: int = max_input_sizes

    def __len__(self):
        return len(self.labels.index)

    def __getitem__(self, idx):
        premise_ids = self.tokenizer.encode(self.labels.iloc[idx].tolist()[0], add_special_tokens=False)

        # -3 to account for the special characters
        self._truncate_seq_pair(premise_ids, self.hypothesis_ids, self.max_input_sizes - 3)

        sequence_token_ids = [
            self.tokenizer.cls_token_id,
            *premise_ids,
            self.tokenizer.sep_token_id,
            *self.hypothesis_ids,
            self.tokenizer.sep_token_id
        ]
        token_type_ids = self._get_token_type(sequence_token_ids, self.tokenizer.sep_token_id)
        attention_mask_ids = [1] * len(sequence_token_ids)

        return torch.tensor(sequence_token_ids), torch.tensor(token_type_ids), torch.tensor(attention_mask_ids)

    @staticmethod
    def _get_token_type(tokens: List[int], sep: int) -> List[int]:
        sep_index = tokens.index(sep) + 1
        return [0] * sep_index + [1] * (len(tokens) - sep_index)

    @staticmethod
    def _truncate_seq_pair(tokens_a: List[int], tokens_b: List[int], max_length: int):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()
