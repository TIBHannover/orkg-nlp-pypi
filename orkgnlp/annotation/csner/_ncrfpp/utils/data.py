from __future__ import print_function
from __future__ import absolute_import

from orkgnlp.annotation.csner._ncrfpp.utils.alphabet import Alphabet
from orkgnlp.annotation.csner._ncrfpp.utils.functions import *
import sys
import orkgnlp.annotation.csner._ncrfpp.utils as utils
# overwrite the import path 'utils' to 'ner/utils' in the pickle file
sys.modules['utils'] = utils

START = '</s>'
UNKNOWN = '</unk>'
PADDING = '</pad>'


class Data:

    def __init__(self):
        self.MAX_SENTENCE_LENGTH = 10000
        self.MAX_WORD_LENGTH = -1
        self.number_normalized = True
        self.norm_word_emb = False
        self.norm_char_emb = False
        self.word_alphabet = Alphabet('word')
        self.char_alphabet = Alphabet('character')

        self.feature_name = []
        self.feature_alphabets = []
        self.feature_num = len(self.feature_alphabets)
        self.feat_config = None

        self.label_alphabet = Alphabet('label', True)
        self.tagScheme = 'NoSeg'  # BMES/BIO
        self.split_token = ' ||| '
        self.seg = True

        # I/O
        self.dset_titles_dir = None
        self.load_titles_model_dir = None
        self.dset_abstracts_dir = None
        self.load_abstracts_model_dir = None

        self.word_emb_dir = None
        self.char_emb_dir = None
        self.feature_emb_dirs = []

        self.raw_texts = []
        self.raw_Ids = []

        self.pretrain_word_embedding = None
        self.pretrain_char_embedding = None
        self.pretrain_feature_embeddings = []

        self.label_size = 0
        self.word_alphabet_size = 0
        self.char_alphabet_size = 0
        self.label_alphabet_size = 0
        self.feature_alphabet_sizes = []
        self.feature_emb_dims = []
        self.norm_feature_embs = []
        self.word_emb_dim = 50
        self.char_emb_dim = 30

        # Networks
        self.word_feature_extractor = 'LSTM'  # "LSTM"/"CNN"/"GRU"/
        self.use_char = True
        self.char_feature_extractor = 'CNN'  # "LSTM"/"CNN"/"GRU"/None
        self.use_crf = True
        self.nbest = 1

        # Training
        self.average_batch_loss = False
        self.optimizer = 'SGD'  # "SGD"/"AdaGrad"/"AdaDelta"/"RMSProp"/"Adam"
        self.status = 'decode'

        # Hyperparameters
        self.HP_cnn_layer = 4
        self.HP_iteration = 200
        self.HP_batch_size = 10
        self.HP_char_hidden_dim = 50
        self.HP_hidden_dim = 200
        self.HP_dropout = 0.5
        self.HP_lstm_layer = 2
        self.HP_bilstm = True

        self.HP_gpu = False
        self.HP_lr = 0.015
        self.HP_lr_decay = 0.05
        self.HP_clip = None
        self.HP_momentum = 0
        self.HP_l2 = 1e-8

    def fix_alphabet(self):
        self.word_alphabet.close()
        self.char_alphabet.close()
        self.label_alphabet.close()
        for idx in range(self.feature_num):
            self.feature_alphabets[idx].close()

    def generate_instance(self, txt):
        self.fix_alphabet()
        self.raw_texts, self.raw_Ids = read_instance(txt, self.word_alphabet, self.char_alphabet, self.feature_alphabets, self.label_alphabet, self.number_normalized)

    def get_entities(self, predict_results):
        sent_num = len(predict_results)
        content_list = self.raw_texts
        assert(sent_num == len(content_list))
        entities = {}
        for idx in range(sent_num):
            sent_length = len(predict_results[idx])
            key = ""
            entity_txt = ""
            for idy in range(sent_length):
                if "S-" in predict_results[idx][idy]:
                    key = predict_results[idx][idy].split("-")[1]
                    if key in entities.keys():
                        entities[key].append(content_list[idx][0][idy])
                    else:
                        entities[key] = [content_list[idx][0][idy]]
                elif "B-" in predict_results[idx][idy]:
                    key = predict_results[idx][idy].split("-")[1]
                    entity_txt = content_list[idx][0][idy].strip()
                elif "I-" in predict_results[idx][idy]:
                    entity_txt += " "+content_list[idx][0][idy].strip()
                elif "E-" in predict_results[idx][idy]:
                    entity_txt += " "+content_list[idx][0][idy].strip()
                    if key in entities.keys():
                        entities[key].append(entity_txt.strip())
                    else:
                        entities[key] = [entity_txt.strip()]
                    key = ""
                    entity_txt = ""

        return entities

    def load(self, data_object):
        self.__dict__.update(data_object)
        self.update_variables()

    def update_variables(self):
        self.HP_gpu = False   
        self.nbest = 1
