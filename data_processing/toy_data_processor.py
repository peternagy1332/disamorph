import numpy as np
import os


class ToyDataProcessor(object):
    def __init__(self, model_configuration):
        self.__config = model_configuration

        self.vocabulary, self.inverse_vocabulary = self.__read_features_to_vocabularies(os.path.join('data', 'toy_vocabulary.txt'))

    def __read_features_to_vocabularies(self, toy_vocabulary):
        print('def __read_features_to_vocabularies(self, file_tags, file_roots):')
        features = []
        with open(toy_vocabulary, encoding='utf-8') as f: features.extend(f.read().splitlines())
        vocabulary = dict(zip(features, range(self.__config.vocabulary_start_index, len(features) + self.__config.vocabulary_start_index)))

        inverse_vocabulary = {v: k for k, v in vocabulary.items()}

        inverse_vocabulary[self.__config.marker_unknown] = '<UNK>'
        inverse_vocabulary[self.__config.marker_padding] = '<PAD>'
        inverse_vocabulary[self.__config.marker_end_of_sentence] = '<EOS>'
        inverse_vocabulary[self.__config.marker_start_of_sentence] = '<SOS>'
        inverse_vocabulary[self.__config.marker_analysis_divider] = '<DIV>'
        inverse_vocabulary[self.__config.marker_go] = '<GO>'

        return vocabulary, inverse_vocabulary


    def get_batches(self):
        print('def get_batches(self):')

        source_input_examples = []
        target_input_examples = []
        target_output_examples = []

        target_repeat = 6

        for word, word_id in self.vocabulary.items():
            source_input_example = [word_id] + [self.__config.marker_end_of_sentence] + [self.__config.marker_padding]*(self.__config.max_source_sequence_length-2)
            target_input_example = [self.__config.marker_go] + [word_id]*target_repeat + [self.__config.marker_padding]*(self.__config.max_target_sequence_length-(1+target_repeat))
            target_output_example = [word_id]*target_repeat + [self.__config.marker_end_of_sentence] + [self.__config.marker_padding]*(self.__config.max_target_sequence_length-(1+target_repeat))

            source_input_examples.append(source_input_example)
            target_input_examples.append(target_input_example)
            target_output_examples.append(target_output_example)


        source_input_examples = np.matrix(source_input_examples)
        target_input_examples = np.matrix(target_input_examples)
        target_output_examples = np.matrix(target_output_examples)

        if self.__config.batch_size>len(source_input_examples):
            raise ValueError("batch_size ("+str(self.__config.batch_size)+") > #{examples}="+str(len(source_input_examples)))

        return source_input_examples, target_input_examples, target_output_examples