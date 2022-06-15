""" CS-NER service decoder. """

from overrides import overrides

from orkgnlp.common.service.base import ORKGNLPBaseDecoder


class CSNerDecoder(ORKGNLPBaseDecoder):
    """
    The CSNerDecoder decodes the CS-NER service model's output
    to a user-friendly one.
    """

    def __init__(self, alphabet):
        """

        :param alphabet: Dict representing the word, char and label alphabets.
        :type alphabet: Dict[str, Dict[str, int]]
        """
        super().__init__()

        self._alphabet = alphabet
        self._UNKNOWN = '</unk>'

    @overrides(check_signature=False)
    def decode(self, model_output, raw_texts, recover, **kwargs):
        predicted_results = []

        for i, batch in enumerate(model_output):
            _, nbest_tag_seq = batch
            tag_seq = nbest_tag_seq[:, :, 0]
            pred_label, _ = self._recover_label(tag_seq, recover[i][0], recover[i][0], recover[i][2])
            predicted_results += pred_label

        entities = self._get_entities(raw_texts, predicted_results)
        return self._prepare_annotations(entities)

    def _recover_label(self, pred_variable, gold_variable, mask_variable, word_recover):
        """
            input:
                pred_variable (batch_size, sent_len): pred tag result
                gold_variable (batch_size, sent_len): gold result variable
                mask_variable (batch_size, sent_len): mask variable
        """
        pred_variable = pred_variable[word_recover]
        gold_variable = gold_variable[word_recover]
        mask_variable = mask_variable[word_recover]
        batch_size = gold_variable.size(0)
        seq_len = gold_variable.size(1)
        mask = mask_variable.cpu().data.numpy()
        pred_tag = pred_variable.cpu().data.numpy()
        gold_tag = gold_variable.cpu().data.numpy()
        batch_size = mask.shape[0]
        pred_label = []
        gold_label = []

        for idx in range(batch_size):
            labels = list(self._alphabet['label'].keys())

            pred = [self._get_instance(labels, pred_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
            gold = [self._get_instance(labels, gold_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]

            pred_label.append(pred)
            gold_label.append(gold)

        return pred_label, gold_label

    @staticmethod
    def _get_entities(raw_texts, predicted_results):
        entities = {}

        for idx in range(len(predicted_results)):
            sent_length = len(predicted_results[idx])
            key = ''
            entity_txt = ''

            for idy in range(sent_length):

                if 'S-' in predicted_results[idx][idy]:
                    key = predicted_results[idx][idy].split('-')[1]

                    if key in entities.keys():
                        entities[key].append(raw_texts[idx][0][idy])
                    else:
                        entities[key] = [raw_texts[idx][0][idy]]

                elif 'B-' in predicted_results[idx][idy]:
                    key = predicted_results[idx][idy].split('-')[1]
                    entity_txt = raw_texts[idx][0][idy].strip()

                elif 'I-' in predicted_results[idx][idy]:
                    entity_txt += ' ' + raw_texts[idx][0][idy].strip()

                elif 'E-' in predicted_results[idx][idy]:
                    entity_txt += ' ' + raw_texts[idx][0][idy].strip()

                    if key in entities.keys():
                        entities[key].append(entity_txt.strip())
                    else:
                        entities[key] = [entity_txt.strip()]

                    key = ''
                    entity_txt = ''

        return entities

    @staticmethod
    def _get_instance(instances, index):
        if index == 0:
            return instances[0]
        try:
            return instances[index - 1]
        except IndexError:
            print('WARNING:Alphabet get_instance, unknown instance, return the first label.')
            return instances[0]

    @staticmethod
    def _prepare_annotations(entities):
        annotations = []

        for concept in entities:
            annotations.append({
                'concept': concept,
                'entities': entities[concept]
            })

        return annotations
