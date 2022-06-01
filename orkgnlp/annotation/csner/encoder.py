""" CS-NER service encoder. """

import sys
import spacy
import torch
import numpy as np
from overrides import overrides

from orkgnlp.common.service.base import ORKGNLPBaseEncoder


class CSNerEncoder(ORKGNLPBaseEncoder):
    """
    The CSNerEncoder encodes the given input to the arguments
    needed to execute a Sequence labeling TorchScript model.
    """

    def __init__(self, alphabet):
        """

        :param alphabet: Dict representing the word, char and label alphabets.
        :type alphabet: Dict[str, Dict[str, int]]
        """
        super().__init__()

        self._alphabet = alphabet
        self._UNKNOWN = '</unk>'
        self._spacy_nlp = spacy.load(
            'en_core_web_md', disable=['tokenizer', 'tagger', 'ner', 'textcat', 'lemmatizer']
        )

    @overrides
    def encode(self, raw_input, **kwargs):
        raw_texts, raw_ids = self._read_instance(raw_input)

        batch_size = 10
        total_batch = len(raw_ids) // batch_size + 1
        model_input = []
        decoder_input = {
            'raw_texts': raw_texts,
            'recover': []
        }

        for batch_id in range(total_batch):
            start = batch_id * batch_size
            end = (batch_id + 1) * batch_size

            if end > len(raw_ids):
                end = len(raw_ids)

            instance = raw_ids[start:end]
            if not instance:
                continue

            word, features, word_len, word_recover, char, char_len, char_recover, label, mask = self._batchify(instance)
            model_input.append((word, features, word_len, char, char_recover, mask, 1, ))
            decoder_input['recover'].append((label, mask, word_recover))

        kwargs.update(decoder_input)
        return model_input, kwargs

    def _read_instance(self, q, char_padding_size=-1, char_padding_symbol='</pad>'):

        document = self._spacy_nlp(q)
        instance_texts = []
        instance_ids = []

        # for sequence labeling text format
        for span in document.sents:

            words, features, chars, labels, word_ids, feature_ids, char_ids, label_ids = [], [], [], [], [], [], [], []
            sentence = [document[i] for i in range(span.start, span.end)]

            for token in sentence:
                word = token.text.strip()

                if not word:
                    continue

                if sys.version_info[0] < 3:
                    word = word.decode('utf-8')

                words.append(word)

                if self._alphabet['number_normalized']:
                    word = self._normalize_word(word)

                label = 'O'
                labels.append(label)
                word_ids.append(self._alphabet['word'].get(word) or self._alphabet['word'].get(self._UNKNOWN))
                label_ids.append(self._alphabet['label'].get(label) or self._alphabet['label'].get(self._UNKNOWN))

                # get features
                feat_list = []
                feat_Id = []

                features.append(feat_list)
                feature_ids.append(feat_Id)

                # get char
                char_list = []
                char_id = []
                for char in word:
                    char_list.append(char)

                if char_padding_size > 0:
                    char_number = len(char_list)
                    if char_number < char_padding_size:
                        char_list = char_list + [char_padding_symbol] * (char_padding_size - char_number)
                    assert (len(char_list) == char_padding_size)
                else:
                    # not padding
                    pass

                for char in char_list:
                    char_id.append(self._alphabet['char'].get(char) or self._alphabet['char'].get(self._UNKNOWN))

                chars.append(char_list)
                char_ids.append(char_id)

            instance_texts.append([words, features, chars, labels])
            instance_ids.append([word_ids, feature_ids, char_ids, label_ids])

        return instance_texts, instance_ids

    @staticmethod
    def _batchify(input_batch_list, if_train=False):
        """
            input: list of words, chars and labels, various length. [[words, features, chars, labels],[words, features, chars,labels],...]
                words: word ids for one sentence. (batch_size, sent_len)
                features: features ids for one sentence. (batch_size, sent_len, feature_num)
                chars: char ids for on sentences, various length. (batch_size, sent_len, each_word_length)
                labels: label ids for one sentence. (batch_size, sent_len)

            output:
                zero padding for word and char, with their batch length
                word_seq_tensor: (batch_size, max_sent_len) Variable
                feature_seq_tensors: [(batch_size, max_sent_len),...] list of Variable
                word_seq_lengths: (batch_size,1) Tensor
                char_seq_tensor: (batch_size*max_sent_len, max_word_len) Variable
                char_seq_lengths: (batch_size*max_sent_len,1) Tensor
                char_seq_recover: (batch_size*max_sent_len,1)  recover char sequence order
                label_seq_tensor: (batch_size, max_sent_len)
                mask: (batch_size, max_sent_len)
        """
        batch_size = len(input_batch_list)
        words = [sent[0] for sent in input_batch_list]
        features = [np.asarray(sent[1]) for sent in input_batch_list]
        feature_num = len(features[0][0])
        chars = [sent[2] for sent in input_batch_list]
        labels = [sent[3] for sent in input_batch_list]
        word_seq_lengths = torch.LongTensor(list(map(len, words)))
        max_seq_len = word_seq_lengths.max().item()
        word_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).long()
        label_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).long()
        feature_seq_tensors = []
        for idx in range(feature_num):
            feature_seq_tensors.append(torch.zeros((batch_size, max_seq_len), requires_grad=if_train).long())
        mask = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).bool()
        for idx, (seq, label, seqlen) in enumerate(zip(words, labels, word_seq_lengths)):
            seqlen = seqlen.item()
            word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
            label_seq_tensor[idx, :seqlen] = torch.LongTensor(label)
            mask[idx, :seqlen] = torch.Tensor([1] * seqlen)
            for idy in range(feature_num):
                feature_seq_tensors[idy][idx, :seqlen] = torch.LongTensor(features[idx][:, idy])
        word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
        word_seq_tensor = word_seq_tensor[word_perm_idx]
        for idx in range(feature_num):
            feature_seq_tensors[idx] = feature_seq_tensors[idx][word_perm_idx]

        label_seq_tensor = label_seq_tensor[word_perm_idx]
        mask = mask[word_perm_idx]
        # deal with char
        # pad_chars (batch_size, max_seq_len)
        pad_chars = [chars[idx] + [[0]] * (max_seq_len - len(chars[idx])) for idx in range(len(chars))]
        length_list = [list(map(len, pad_char)) for pad_char in pad_chars]
        max_word_len = max(map(max, length_list))
        char_seq_tensor = torch.zeros((batch_size, max_seq_len, max_word_len), requires_grad=if_train).long()
        char_seq_lengths = torch.LongTensor(length_list)
        for idx, (seq, seqlen) in enumerate(zip(pad_chars, char_seq_lengths)):
            for idy, (word, wordlen) in enumerate(zip(seq, seqlen)):
                # print len(word), wordlen
                char_seq_tensor[idx, idy, :wordlen] = torch.LongTensor(word)

        char_seq_tensor = char_seq_tensor[word_perm_idx].view(batch_size * max_seq_len, -1)
        char_seq_lengths = char_seq_lengths[word_perm_idx].view(batch_size * max_seq_len, )
        char_seq_lengths, char_perm_idx = char_seq_lengths.sort(0, descending=True)
        char_seq_tensor = char_seq_tensor[char_perm_idx]
        _, char_seq_recover = char_perm_idx.sort(0, descending=False)
        _, word_seq_recover = word_perm_idx.sort(0, descending=False)
        return word_seq_tensor, torch.tensor(feature_seq_tensors), word_seq_lengths, word_seq_recover, char_seq_tensor, char_seq_lengths, char_seq_recover, label_seq_tensor, mask

    @staticmethod
    def _normalize_word(word):
        new_word = ''

        for char in word:

            if char.isdigit():
                new_word += '0'
            else:
                new_word += char

        return new_word
