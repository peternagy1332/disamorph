import csv
import glob
from collections import namedtuple

import pandas as pd
import numpy as np
import itertools

from data_processing.analyses_processor import AnalysesProcessor
from utils import Utils


class DataProcessor(object):
    def __init__(self, model_configuration):
        self.__config = model_configuration

        self.vocabulary, self.inverse_vocabulary = self.__read_features_to_vocabularies(self.__config.train_files_tags, self.__config.train_files_roots)

    def __read_features_to_vocabularies(self, file_tags, file_roots):
        print('def __read_features_to_vocabularies(self, file_tags, file_roots):')
        features = []
        with open(file_tags, encoding='utf-8') as f: features.extend(f.read().splitlines())
        with open(file_roots, encoding='utf-8') as f: features.extend(f.read().splitlines())
        vocabulary = dict(zip(features, range(self.__config.vocabulary_start_index, len(features) + self.__config.vocabulary_start_index)))

        inverse_vocabulary = {v: k for k, v in vocabulary.items()}

        inverse_vocabulary[self.__config.marker_unknown] = '<UNK>'
        inverse_vocabulary[self.__config.marker_padding] = '<PAD>'
        inverse_vocabulary[self.__config.marker_end_of_sentence] = '<EOS>'
        inverse_vocabulary[self.__config.marker_start_of_sentence] = '<SOS>'
        inverse_vocabulary[self.__config.marker_analysis_divider] = '<DIV>'
        
        return vocabulary, inverse_vocabulary

    def get_sentence_dataframes(self):
        print('def get_sentence_dataframes(self):')
        analyses_processor = AnalysesProcessor(self.__config, self.vocabulary)

        sentence_dataframes = []

        read_rows = 0
        should_stop = False
        for file in glob.glob(self.__config.train_files_corpus):
            if should_stop:
                break

            with open(file, newline='', encoding='utf8') as csvfile:
                csvreader = csv.reader(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE)

                sentence_dict = {'word': [], 'correct_analysis_vector': [], 'correct_analysis': [], 'root_id': []}
                for i in range(self.__config.window_length - 1):
                    sentence_dict['word'].append(self.inverse_vocabulary[self.__config.marker_start_of_sentence])
                    sentence_dict['correct_analysis_vector'].append([self.__config.marker_start_of_sentence])
                    sentence_dict['correct_analysis'].append(self.inverse_vocabulary[self.__config.marker_start_of_sentence])
                    sentence_dict['root_id'].append([self.__config.marker_start_of_sentence])

                for row in csvreader:
                    # End of sentence
                    if len(row) == 0:
                        sentence_dataframes.append(pd.DataFrame.from_dict(sentence_dict))
                        sentence_dict = {'word': [], 'correct_analysis_vector': [], 'correct_analysis': [], 'root_id': []}
                        for i in range(self.__config.window_length - 1):
                            sentence_dict['word'].append(self.inverse_vocabulary[self.__config.marker_start_of_sentence])
                            sentence_dict['correct_analysis_vector'].append([self.__config.marker_start_of_sentence])
                            sentence_dict['correct_analysis'].append(self.inverse_vocabulary[self.__config.marker_start_of_sentence])
                            sentence_dict['root_id'].append([self.__config.marker_start_of_sentence])
                        continue

                    sentence_dict['word'].append(row[0])
                    sentence_dict['correct_analysis_vector'].append(analyses_processor.lookup_analysis_to_list(row[4]))
                    sentence_dict['correct_analysis'].append(row[4])
                    sentence_dict['root_id'].append(analyses_processor.lookup_analysis_to_list(analyses_processor.get_root_from_analysis(row[4])))

                    read_rows+=1
                    if self.__config.rows_to_read_num is not None and read_rows>=self.__config.rows_to_read_num:
                        should_stop = True
                        break

        print('\t#{read rows}: ', read_rows)
        print('\t#{sentences}: ',len(sentence_dataframes))
        return sentence_dataframes

    def format_window_word_analyses(self, combinations_in_window, EOS_needed=True):
        flattened_markered_combinations = []

        for combination in combinations_in_window:
            combination = list(combination)

            analyses_divided = [[self.__config.marker_analysis_divider]] * (len(combination) * 2 - 1)
            analyses_divided[0::2] = combination
            flattened_list = list(itertools.chain.from_iterable(analyses_divided))

            if EOS_needed:
                EOS_appended = flattened_list + [self.__config.marker_end_of_sentence]
                flattened_markered_combinations.append(EOS_appended)
            else:
                flattened_markered_combinations.append(flattened_list)

        return flattened_markered_combinations

    def pad_batch(self, batch_list, max_sequence_length, min_batch_size = None):
        horizontal_pad = list(map(
            lambda sequence: sequence + [self.__config.marker_padding] * (max_sequence_length - len(sequence)),
            batch_list))

        # Vertical padding
        if min_batch_size is not None:
            padding_row = [self.__config.marker_padding] * max_sequence_length

            for i in range(min_batch_size - len(horizontal_pad)):
                horizontal_pad.append(padding_row)

        return horizontal_pad

    def train_dataframes_to_batches(self, train_dataframes):
        print('def train_dataframes_to_batches(self, train_dataframes, test_dataframes):')

        source_input_examples = []
        target_input_examples = []
        target_output_examples = []

        for sentence_dataframe in train_dataframes:
            sentence_source_input_examples,\
            sentence_target_input_examples,\
            sentence_target_output_examples = self.__sentence_dataframe_to_examples(sentence_dataframe)

            source_input_examples.extend(sentence_source_input_examples)
            target_input_examples.extend(sentence_target_input_examples)
            target_output_examples.extend(sentence_target_output_examples)

        source_input_examples = np.matrix(source_input_examples)
        target_input_examples = np.matrix(target_input_examples)
        target_output_examples = np.matrix(target_output_examples)

        if self.__config.batch_size>len(source_input_examples):
            raise ValueError("batch_size ("+str(self.__config.batch_size)+") > #{examples}="+str(len(source_input_examples)))

        TrainBatches = namedtuple('Dataset', ['source_input_batches', 'target_input_batches', 'target_output_batches'])

        train_batches = TrainBatches(
            source_input_batches=[source_input_examples[i:i + self.__config.batch_size] for i in range(source_input_examples.shape[0] // self.__config.batch_size)],
            target_input_batches=[target_input_examples[i:i + self.__config.batch_size] for i in range(target_input_examples.shape[0] // self.__config.batch_size)],
            target_output_batches=[target_output_examples[i:i + self.__config.batch_size] for i in range(target_output_examples.shape[0] // self.__config.batch_size)]
        )

        print('\t#{batches}: ', len(train_batches.source_input_batches))
        print('\t#{examples}: ', len(train_batches.source_input_batches)*self.__config.batch_size)

        return train_batches

    def __sentence_dataframe_to_examples(self, sentence_dataframe):
        # print('def __sentence_dataframe_to_examples(self, sentence_dataframe):')
        source_input_sequences = []
        target_input_sequences = []
        target_output_sequences = []

        for analysis_id in range(sentence_dataframe.shape[0] - self.__config.window_length):
            source_window_dataframe = sentence_dataframe.loc[analysis_id:analysis_id + self.__config.window_length - 1]
            target_sequence = sentence_dataframe.loc[analysis_id+self.__config.window_length]
            source_sequence = self.format_window_word_analyses([source_window_dataframe['correct_analysis_vector'].tolist() + [target_sequence['root_id']]], False)
            target_input_sequence = self.format_window_word_analyses([[[self.__config.marker_start_of_sentence] + target_sequence['correct_analysis_vector']]], False)
            target_output_sequence = self.format_window_word_analyses([[target_sequence['correct_analysis_vector']]])

            source_input_sequences.append(source_sequence[0])
            target_input_sequences.append(target_input_sequence[0])
            target_output_sequences.append(target_output_sequence[0])

        source_input_examples = self.pad_batch(source_input_sequences, self.__config.max_source_sequence_length)
        target_input_examples = self.pad_batch(target_input_sequences, self.__config.max_target_sequence_length)
        target_output_examples = self.pad_batch(target_output_sequences, self.__config.max_target_sequence_length)

        return source_input_examples, target_input_examples, target_output_examples