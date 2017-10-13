import glob
from collections import namedtuple

import pandas as pd
import numpy as np
import re

import yaml

from data_processing.analyses_processor import AnalysesProcessor


class TrainDataProcessor(object):
    def __init__(self, model_configuration):
        self.__config = model_configuration

        self.vocabulary, self.inverse_vocabulary = self.__read_features_to_vocabularies(
            self.__config.train_files_tags,
            self.__config.train_files_roots)

        self.corpus_dataframe = self.__read_corpus_dataframe(self.__config.train_files_corpus)

    def __read_features_to_vocabularies(self, file_tags, file_roots):
        features = []
        with open(file_tags, encoding='utf-8') as f: features.extend(f.read().splitlines())
        with open(file_roots, encoding='utf-8') as f: features.extend(f.read().splitlines())
        vocabulary = dict(zip(features, range(self.__config.vocabulary_start_index,
                                        len(features) + self.__config.vocabulary_start_index)))

        inverse_vocabulary = {v: k for k, v in vocabulary.items()}

        inverse_vocabulary[self.__config.marker_unknown] = '<UNK>'
        inverse_vocabulary[self.__config.marker_padding] = '<PAD>'
        inverse_vocabulary[self.__config.marker_end_of_sentence] = '<EOS>'
        inverse_vocabulary[self.__config.marker_start_of_sentence] = '<SOS>'
        inverse_vocabulary[self.__config.marker_analysis_divider] = '<DIV>'
        
        return vocabulary, inverse_vocabulary

    def __read_corpus_dataframe(self, path_corpuses):
        analyses_processor = AnalysesProcessor(self.__config, self.vocabulary)
        corpus_dataframe = pd.concat(
            (
                pd.read_csv(f,
                            sep='\t',
                            usecols=[0, 4],
                            skip_blank_lines=False, # WARNING: use .dropna() when constructing batches
                            header=None,
                            nrows=self.__config.nrows,
                            names=['word', 'correct_analysis'])
                for f in glob.glob(path_corpuses)), ignore_index=True
        )

        corpus_dataframe['correct_analysis'] = corpus_dataframe['correct_analysis'].apply(analyses_processor.lookup_analysis_to_list)

        return corpus_dataframe


    def process_dataset(self):

        source_sequences = []
        target_sequences = []

        max_source_sequence_length = 0
        max_target_sequence_length = 0

        nonempty_corpus_dataframe = self.corpus_dataframe.dropna()

        # Gathering examples in list format
        for analysis_id in range(nonempty_corpus_dataframe.shape[0] - self.__config.window_length):
            source_sequence = []

            source_window_dataframe = nonempty_corpus_dataframe.loc[
                                      analysis_id:analysis_id + self.__config.window_length - 1]

            for source_row in source_window_dataframe['correct_analysis']:
                source_sequence.extend(source_row + [self.__config.marker_analysis_divider])

            target_sequence = nonempty_corpus_dataframe.loc[analysis_id + self.__config.window_length]['correct_analysis']

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
        target_input_matrix[:, 0] = np.full((target_input_matrix.shape[0], 1),
                                            self.__config.marker_start_of_sentence, dtype=np.int32)

        Dataset = namedtuple('Dataset', ['source_input_batches', 'target_input_batches', 'target_output_batches'])

        dataset = Dataset(
            source_input_batches=[source_input_matrix[i:i + self.__config.batch_size] for i in
                                      range(source_input_matrix.shape[0] // self.__config.batch_size)],
            target_input_batches=[target_input_matrix[i:i + self.__config.batch_size] for i in
                                      range(target_input_matrix.shape[0] // self.__config.batch_size)],
            target_output_batches=[target_output_matrix[i:i + self.__config.batch_size] for i in
                                       range(target_output_matrix.shape[0] // self.__config.batch_size)]
        )

        print(dataset.source_input_batches[0])
        print(dataset.target_input_batches[0])
        print(dataset.target_output_batches[0])

        DatasetMetadata = namedtuple('DatasetMetadata', ['max_source_sequence_length', 'max_target_sequence_length'])

        self.__dataset_metadata = DatasetMetadata(
            max_source_sequence_length=max_source_sequence_length,
            max_target_sequence_length=max_target_sequence_length
        )

        return dataset, self.__dataset_metadata

    def save_dataset_metadata(self):
        print('Saving dataset metadata to ', self.__config.train_files_dataset_metadata)

        dataset_metadata = dict(
            max_source_sequence_length = self.__dataset_metadata.max_source_sequence_length,
            max_target_sequence_length = self.__dataset_metadata.max_target_sequence_length
        )

        with open(self.__config.train_files_dataset_metadata, 'w', encoding='utf8') as outfile:
            yaml.dump(dataset_metadata, outfile, default_flow_style=False)

    def load_dataset_metadata(self):
        print('Loading dataset metadata from ', self.__config.train_files_dataset_metadata)

        with open(self.__config.train_files_dataset_metadata, 'r', encoding='utf8') as infile:
            try:
                loaded_dict = yaml.load(infile)
                return namedtuple('DatasetMetadata', loaded_dict.keys())(**loaded_dict)
            except yaml.YAMLError as e:
                print(e)