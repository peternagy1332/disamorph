import glob
from collections import namedtuple

import pandas as pd
import numpy as np
import sys

import yaml

from data_processing.analyses_processor import AnalysesProcessor
from utils import update_progress


class TrainDataProcessor(object):
    def __init__(self, model_configuration):
        self.__config = model_configuration

        self.vocabulary, self.inverse_vocabulary = self.__read_features_to_vocabularies(
            self.__config.train_files_tags,
            self.__config.train_files_roots)

    def __read_features_to_vocabularies(self, file_tags, file_roots):
        print('def __read_features_to_vocabularies(self, file_tags, file_roots):')
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

    def read_corpus_dataframe(self):
        print('def __read_corpus_dataframe(self, path_corpuses):')
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
                for f in glob.glob(self.__config.train_files_corpus)), ignore_index=True
        )

        corpus_dataframe['correct_analysis'] = corpus_dataframe['correct_analysis']\
            .apply(analyses_processor.lookup_analysis_to_list)

        corpus_dataframe = self.__insert_start_of_sentence_rows(corpus_dataframe)

        self.corpus_dataframe = corpus_dataframe

    def __insert_start_of_sentence_rows(self, corpus_dataframe):
        print('def __insert_start_of_sentence_rows(self, corpus_dataframe):')
        df = pd.DataFrame(columns=('word', 'correct_analysis'))
        df_index = 0
        for i in range(self.__config.window_length-1):
            df.loc[df_index] = [self.inverse_vocabulary[self.__config.marker_start_of_sentence], [self.__config.marker_start_of_sentence]]
            df_index+=1

        corpus_dataframe_rows = len(corpus_dataframe)

        for index, row in corpus_dataframe.iterrows():
            if pd.isnull(row['word']):
                for i in range(self.__config.window_length - 1):
                    df.loc[df_index] = [self.inverse_vocabulary[self.__config.marker_start_of_sentence], [self.__config.marker_start_of_sentence]]
                    df_index += 1
                continue

            df.loc[df_index] = [row['word'], row['correct_analysis']]
            df_index+=1

            update_progress(index, corpus_dataframe_rows, 'Inserting <SOS> (percentage approximated)')

        return df



    def process_dataset(self):
        print('\ndef process_dataset(self):')

        source_sequences = []
        target_sequences = []

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

            # Lists constructed
            source_sequences.append(source_sequence)
            target_sequences.append(target_sequence)

        # Padding lists
        for i, source_sequence in enumerate(source_sequences):
            source_sequences[i] = np.lib.pad(source_sequence,
                                             (0, self.__config.max_source_sequence_length - len(source_sequence)),
                                             'constant', constant_values=self.__config.marker_padding)

        for i, target_sequence in enumerate(target_sequences):
            target_sequences[i] = np.lib.pad(target_sequence,
                                             (0, self.__config.max_target_sequence_length - len(target_sequence)),
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

        return dataset