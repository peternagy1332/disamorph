from collections import namedtuple
import numpy as np
import yaml


class TrainDataProcessor(object):
    def __init__(self, model_configuration):
        self.__config = model_configuration

        self.vocabulary, self.inverse_vocabulary = self.__read_features_to_vocabularies('data/toy_vocabulary.txt')

    def __read_features_to_vocabularies(self, file):
        features = []
        with open(file, encoding='utf-8') as f: features.extend(f.read().splitlines())
        vocabulary = dict(zip(features,
                              range(
                                  self.__config.vocabulary_start_index,
                                  len(features) + self.__config.vocabulary_start_index)))

        inverse_vocabulary = {v: k for k, v in vocabulary.items()}

        inverse_vocabulary[self.__config.marker_unknown] = '<UNK>'
        inverse_vocabulary[self.__config.marker_padding] = '<PAD>'
        inverse_vocabulary[self.__config.marker_end_of_sentence] = '<EOS>'
        inverse_vocabulary[self.__config.marker_start_of_sentence] = '<SOS>'
        inverse_vocabulary[self.__config.marker_analysis_divider] = '<DIV>'

        return vocabulary, inverse_vocabulary

    def process_dataset(self):

        source_sequences = [[word_id] + [self.__config.marker_end_of_sentence] + [self.__config.marker_padding]*5 for word_id in self.vocabulary.values()]
        target_sequences = [[source_sequence[0]]*3 + [self.__config.marker_end_of_sentence] + [self.__config.marker_padding]*1 for source_sequence in source_sequences]

        source_input_matrix = np.matrix(source_sequences, dtype=np.int32)
        target_output_matrix = np.matrix(target_sequences, dtype=np.int32)
        target_input_matrix = np.roll(target_output_matrix, 1, axis=1)
        target_input_matrix[:, 0] = np.full((target_input_matrix.shape[0], 1), self.__config.marker_start_of_sentence,
                                            dtype=np.int32)

        Dataset = namedtuple('Dataset', ['source_input_batches', 'target_input_batches', 'target_output_batches'])

        dataset = Dataset(
            source_input_batches=[source_input_matrix[i:i + self.__config.batch_size] for i in
                                  range(source_input_matrix.shape[0] // self.__config.batch_size)],
            target_input_batches=[target_input_matrix[i:i + self.__config.batch_size] for i in
                                  range(target_input_matrix.shape[0] // self.__config.batch_size)],
            target_output_batches=[target_output_matrix[i:i + self.__config.batch_size] for i in
                                   range(target_output_matrix.shape[0] // self.__config.batch_size)]
        )

        return dataset