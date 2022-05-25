from __future__ import print_function
from __future__ import absolute_import

import io
import sys
import spacy
import contextlib

with contextlib.redirect_stdout(io.StringIO()):
    spacy.cli.download('en_core_web_md', False, False, '--quiet')
    spacy_nlp = spacy.load(
        'en_core_web_md', disable=['tokenizer', 'tagger', 'ner', 'textcat', 'lemmatizer']
    )


def normalize_word(word):
    new_word = ""
    for char in word:
        if char.isdigit():
            new_word += '0'
        else:
            new_word += char
    return new_word


def read_instance(input_str, word_alphabet, char_alphabet, feature_alphabets, label_alphabet, number_normalized,
                  char_padding_size=-1, char_padding_symbol='</pad>'):
    feature_num = len(feature_alphabets)

    document = spacy_nlp(input_str)

    instance_texts = []
    instance_Ids = []
    words = []
    features = []
    chars = []
    labels = []
    word_Ids = []
    feature_Ids = []
    char_Ids = []
    label_Ids = []

    # for sequence labeling text format
    for span in document.sents:
        sentence = [document[i] for i in range(span.start, span.end)]
        for token in sentence:
            word = token.text.strip()

            if word == "":
                continue

            if sys.version_info[0] < 3:
                word = word.decode('utf-8')

            words.append(word)

            if number_normalized:
                word = normalize_word(word)

            label = "O"
            labels.append(label)
            word_Ids.append(word_alphabet.get_index(word))
            label_Ids.append(label_alphabet.get_index(label))

            # get features
            feat_list = []
            feat_Id = []
            for idx in range(feature_num):
                feat_idx = pairs[idx + 1].split(']', 1)[-1]
                feat_list.append(feat_idx)
                feat_Id.append(feature_alphabets[idx].get_index(feat_idx))

            features.append(feat_list)
            feature_Ids.append(feat_Id)

            # get char
            char_list = []
            char_Id = []
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
                char_Id.append(char_alphabet.get_index(char))
            chars.append(char_list)
            char_Ids.append(char_Id)
        instance_texts.append([words, features, chars, labels])
        instance_Ids.append([word_Ids, feature_Ids, char_Ids, label_Ids])
        words = []
        features = []
        chars = []
        labels = []
        word_Ids = []
        feature_Ids = []
        char_Ids = []
        label_Ids = []

    return instance_texts, instance_Ids
