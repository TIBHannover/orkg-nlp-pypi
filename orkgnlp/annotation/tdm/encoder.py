""" TDM-Extraction service encoder. """

import torch
from overrides import overrides
from torch.utils.data import Dataset, DataLoader
from transformers import XLNetTokenizer

from orkgnlp.common.service.base import ORKGNLPBaseEncoder


class TdmExtractorEncoder(ORKGNLPBaseEncoder):
    """
    The TdmExtractorEncoder encodes the given input to the arguments
    needed to execute an XLNetForSequenceClassification model.
    """

    def __init__(self, labels, batch_size):
        """

        :param labels: TDM gold labels given as one-columned-dataframe
        :type labels: pandas.DataFrame
        :param batch_size: Size of the batches used during model prediction.
        :type batch_size: int
        """
        super().__init__()

        self.labels = labels
        self.batch_size = batch_size
        self.tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
        self.max_input_sizes = self.tokenizer.max_model_input_sizes['xlnet-base-cased'] or 512
        self.device = 'cpu'

    @overrides
    def encode(self, raw_input, **kwargs):
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
    def _collate_batch(batch):
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

    def __init__(self, text, labels, tokenizer, max_input_sizes):
        """

        :param text: Input text (hypothesis) to be concatenated with all known labels (premises).
        :type text: str
        :param labels: TDM gold labels given as one-columned-dataframe
        :type labels: pandas.DataFrame
        :param tokenizer: Tokenizer for tokenizing the texts.
        :type tokenizer: transformers.PreTrainedTokenizer
        :param max_input_sizes: Max length of a sequence including special characters.
        :type max_input_sizes: int
        """
        self.labels = labels
        self.tokenizer = tokenizer
        self.hypothesis_ids = self.tokenizer.encode(text, add_special_tokens=False)
        self.max_input_sizes = max_input_sizes

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
    def _get_token_type(tokens, sep):
        sep_index = tokens.index(sep) + 1
        return [0] * sep_index + [1] * (len(tokens) - sep_index)

    @staticmethod
    def _truncate_seq_pair(tokens_a, tokens_b, max_length):
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
