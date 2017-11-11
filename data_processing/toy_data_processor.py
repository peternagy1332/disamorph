import numpy as np
import os
import random


class ToyDataProcessor(object):
    def __init__(self, model_configuration):
        self.__config = model_configuration

        self.vocabulary, self.inverse_vocabulary = self.__read_features_to_vocabularies(os.path.join('data', 'toy_vocabulary.txt'))

    def __read_features_to_vocabularies(self, toy_vocabulary):
        print('def __read_and_build_vocabulary(self, file_tags, file_roots):')
        features = []
        if self.__config.rows_to_read_num is None:
            with open(toy_vocabulary, encoding='utf-8') as f: features.extend(f.read().splitlines())
        else:
            with open(toy_vocabulary, encoding='utf-8') as f: features.extend(f.read().splitlines()[:self.__config.rows_to_read_num])
        vocabulary = dict(zip(features, range(self.__config.vocabulary_start_index, len(features) + self.__config.vocabulary_start_index)))

        inverse_vocabulary = {v: k for k, v in vocabulary.items()}

        # inverse_vocabulary[self.__config.marker_unknown] = '<UNK>'
        # inverse_vocabulary[self.__config.marker_padding] = '<PAD>'
        # inverse_vocabulary[self.__config.marker_end_of_sentence] = '<EOS>'
        # inverse_vocabulary[self.__config.marker_start_of_sentence] = '<SOS>'
        # inverse_vocabulary[self.__config.marker_analysis_divider] = '<DIV>'
        # inverse_vocabulary[self.__config.marker_go] = '<GO>'

        inverse_vocabulary[self.__config.marker_unknown] = ''
        inverse_vocabulary[self.__config.marker_padding] = ''
        inverse_vocabulary[self.__config.marker_end_of_sentence] = ''
        inverse_vocabulary[self.__config.marker_start_of_sentence] = ''
        inverse_vocabulary[self.__config.marker_analysis_divider] = ''
        inverse_vocabulary[self.__config.marker_go] = ''

        return vocabulary, inverse_vocabulary


    def get_batches(self):
        print('def get_batches(self):')

        source_input_examples = []
        target_input_examples = []
        target_output_examples = []


        for word, word_id in self.vocabulary.items():
            target_repeat = random.randint(3,self.__config.max_target_sequence_length-1)
            target_sequence = [word_id]*target_repeat
            source_input_example = [word_id] + [self.__config.marker_end_of_sentence] + [self.__config.marker_padding]*(self.__config.max_source_sequence_length-2)
            target_input_example = [self.__config.marker_go] + target_sequence + [self.__config.marker_padding]*(self.__config.max_target_sequence_length-(1+target_repeat))
            target_output_example = target_sequence + [self.__config.marker_end_of_sentence] + [self.__config.marker_padding]*(self.__config.max_target_sequence_length-(1+target_repeat))

            source_input_examples.append(source_input_example)
            target_input_examples.append(target_input_example)
            target_output_examples.append(target_output_example)


        source_input_examples = np.matrix(source_input_examples)
        target_input_examples = np.matrix(target_input_examples)
        target_output_examples = np.matrix(target_output_examples)

        print('\t#{examples}:',source_input_examples.shape[0])

        if self.__config.batch_size>source_input_examples.shape[0]:
            raise ValueError("batch_size ("+str(self.__config.batch_size)+") > #{examples}="+str(source_input_examples.shape[0]))

        return source_input_examples, target_input_examples, target_output_examples