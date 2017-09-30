import glob
import pandas as pd
import numpy as np
import re
from config import Dataset

class TrainDataProcessor(object):
    def __init__(self, model_configuration):
        self.__config = model_configuration

        self.__vocabulary = self.__read_features_to_vocabulary(self.__config.train_files_tags,
                                                               self.__config.train_files_roots)

        self.__corpus_dataframe = self.__read_corpus_dataframe(self.__config.train_files_corpus)

    def __read_features_to_vocabulary(self, file_tags, file_roots):
        features = []
        with open(file_tags, encoding='utf-8') as f: features.extend(f.read().splitlines())
        with open(file_roots, encoding='utf-8') as f: features.extend(f.read().splitlines())
        return dict(zip(features, range(self.__config.vocabulary_start_index,
                                        len(features) + self.__config.vocabulary_start_index)))

    def __read_corpus_dataframe(self, path_corpuses):
        return pd.concat(
            (pd.read_csv(f, sep='\t', usecols=[4], skip_blank_lines=True, header=None, nrows=self.__config.nrows,
                         names=['analysis'])
             for f in glob.glob(path_corpuses)), ignore_index=True)\
            .applymap(lambda analysis: self.__lookup_analysis_to_list(analysis))

    def __lookup_analysis_to_list(self, analysis):
        root = re.search(r'\w+', analysis, re.UNICODE).group(0)
        tags = re.findall(r'\[[^]]*\]', analysis, re.UNICODE)
        return [self.__vocabulary.get(root, self.__config.marker_unknown)] + [
            self.__vocabulary.get(tag, self.__config.marker_unknown) for tag in tags]

    def process_dataset(self):

        source_sequences = []
        target_sequences = []

        max_source_sequence_length = 0
        max_target_sequence_length = 0

        # Gathering examples in list format
        for analysis_id in range(self.__corpus_dataframe.shape[0] - self.__config.window_length):
            source_sequence = []

            source_window_dataframe = self.__corpus_dataframe.loc[
                                      analysis_id:analysis_id + self.__config.window_length - 1]

            for source_row in source_window_dataframe['analysis']:
                source_sequence.extend(source_row + [self.__config.marker_analysis_divider])

            target_sequence = self.__corpus_dataframe.loc[analysis_id + self.__config.window_length]['analysis']

            source_sequence.extend([target_sequence[0], self.__config.marker_end_of_sentence])

            if len(source_sequence) > max_source_sequence_length:
                max_source_sequence_length = len(source_sequence)

            if len(target_sequence) > max_target_sequence_length:
                max_target_sequence_length = len(target_sequence)

            # Lists constructed
            source_sequences.append(source_sequence)
            target_sequences.append(target_sequence)

        # Padding lists
        for i, source_sequence in enumerate(source_sequences):
            source_sequences[i] = np.lib.pad(source_sequence,
                                             (0, max_source_sequence_length - len(source_sequence)),
                                             'constant', constant_values=self.__config.marker_padding)

        for i, target_sequence in enumerate(target_sequences):
            target_sequences[i] = np.lib.pad(target_sequence,
                                             (0, max_target_sequence_length - len(target_sequence)),
                                             'constant', constant_values=self.__config.marker_padding)

        source_input_matrix = np.matrix(source_sequences, dtype=np.int32)
        target_output_matrix = np.matrix(target_sequences, dtype=np.int32)
        target_input_matrix = np.roll(target_output_matrix, 1, axis=1)
        target_input_matrix[:, 0] = np.full((target_input_matrix.shape[0], 1), self.__config.marker_start_of_sentence, dtype=np.int32)

        dataset = Dataset()
        
        dataset.source_input_batches=[source_input_matrix[i:i + self.__config.batch_size] for i in
                                      range(source_input_matrix.shape[0] // self.__config.batch_size)]
        
        dataset.target_input_batches=[target_input_matrix[i:i + self.__config.batch_size] for i in
                                      range(target_input_matrix.shape[0] // self.__config.batch_size)]
        
        dataset.target_output_batches=[target_output_matrix[i:i + self.__config.batch_size] for i in
                                       range(target_output_matrix.shape[0] // self.__config.batch_size)]

        return dataset, max_source_sequence_length, max_target_sequence_length, self.__vocabulary
