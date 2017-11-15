import csv
import glob
import os
from collections import namedtuple

import pandas as pd
import numpy as np
import itertools
import sys
from random import shuffle

class DataProcessor(object):
    def __init__(self, model_configuration, analyses_processor):
        self.__config = model_configuration
        self.__analyses_processor = analyses_processor

        self.__source_input_examples = None
        self.__target_input_examples = None
        self.__target_output_examples = None

        self.__sentence_source_input_examples_matrices = []
        self.__sentence_target_input_examples_matrices = []
        self.__sentence_target_output_examples_matrices = []


    def get_sentence_dicts(self, use_saved_matrices=True):
        print('def get_sentence_dicts(self, use_saved_matrices='+str(use_saved_matrices)+'):')
        if os.path.isdir(self.__config.data_train_matrices) and os.listdir(os.path.join(self.__config.data_train_matrices, 'source_input'))!=[] and use_saved_matrices:
            print('\tUsing previously saved train matrices. Returning no sentences.')
            return []

        sentence_dicts = []

        read_rows = 0
        should_stop = False
        for file in glob.glob(self.__config.data_train_dataset):
            if should_stop:
                break

            with open(file, newline='', encoding='utf8') as csvfile:
                csvreader = csv.reader(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE)

                sentence_dict = {'word': [], 'correct_analysis_vector': [], 'correct_analysis': [] }
                for i in range(self.__config.network_window_length - 1):
                    sentence_dict['word'].append(self.__analyses_processor.inverse_vocabulary[self.__config.marker_start_of_sentence])
                    sentence_dict['correct_analysis_vector'].append([self.__config.marker_start_of_sentence])
                    sentence_dict['correct_analysis'].append(self.__analyses_processor.inverse_vocabulary[self.__config.marker_start_of_sentence])

                csv.field_size_limit(sys.maxsize)
                for row in csvreader:
                    # End of sentence
                    if len(row) == 0:
                        sentence_dicts.append(sentence_dict)
                        sentence_dict = {'word': [], 'correct_analysis_vector': [], 'correct_analysis': [] }
                        for i in range(self.__config.network_window_length - 1):
                            sentence_dict['word'].append(self.__analyses_processor.inverse_vocabulary[self.__config.marker_start_of_sentence])
                            sentence_dict['correct_analysis_vector'].append([self.__config.marker_start_of_sentence])
                            sentence_dict['correct_analysis'].append(self.__analyses_processor.inverse_vocabulary[self.__config.marker_start_of_sentence])
                        continue

                    sentence_dict['word'].append(row[0])

                    # No correct analysis available
                    if '+?' == row[8][-2:]:
                        if self.__config.data_example_resolution == 'morpheme':
                            self.__analyses_processor.extend_vocabulary(row[8])
                            # Actually no tags
                            features_and_tags = [row[8]]
                        elif self.__config.data_example_resolution == 'character':
                            # Actually no tags
                            features_and_tags = list(row[8][:-2])
                            for character in features_and_tags:
                                self.__analyses_processor.extend_vocabulary(character)
                    else:
                        tape = self.__analyses_processor.raw_tape_to_tape(row[8])
                        if self.__config.data_example_resolution == 'morpheme':
                            features_and_tags = self.__analyses_processor.tape_to_morphemes_and_tags(tape)
                        elif self.__config.data_example_resolution == 'character':
                            features_and_tags = tape

                    sentence_dict['correct_analysis_vector'].append(self.__analyses_processor.lookup_morphemes_and_tags_to_ids(features_and_tags))
                    sentence_dict['correct_analysis'].append("".join(features_and_tags))


                    read_rows+=1

                    if self.__config.data_sentences_to_read_num is not None and len(sentence_dicts)>=self.__config.data_sentences_to_read_num:
                        should_stop = True
                        break

        self.__analyses_processor.write_vocabulary_file()

        print('\t#{total sentences}:', len(sentence_dicts))
        print('\t#{read rows}:', read_rows)

        return sentence_dicts

    def get_split_sentence_dicts(self):
        sentence_dicts = self.get_sentence_dicts()
        # Splitting sentence dict list to train, validation, test
        train_sentence_dicts = sentence_dicts[:int(len(sentence_dicts) * self.__config.data_train_ratio)]

        validation_sentence_dicts = sentence_dicts[
                                    int(len(sentence_dicts) * self.__config.data_train_ratio):
                                    int(len(sentence_dicts) * (self.__config.data_train_ratio + self.__config.data_validation_ratio))
                                    ]

        test_sentence_dicts = sentence_dicts[int(len(sentence_dicts) * (self.__config.data_train_ratio + self.__config.data_validation_ratio)):]


        print('\t#{train sentences}:', len(train_sentence_dicts))
        print('\t#{validation sentences}:', len(validation_sentence_dicts))
        print('\t#{test sentences}:', len(test_sentence_dicts))

        if len(train_sentence_dicts) + len(validation_sentence_dicts) + len(test_sentence_dicts) != len(sentence_dicts):
            raise ValueError('SUM len of train, validation, test sentences != len of sentences')

        return train_sentence_dicts, validation_sentence_dicts, test_sentence_dicts

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

    def pad_batch(self, batch_list, max_sequence_length, min_train_batch_size = None):
        horizontal_pad = list(map(lambda sequence: sequence + [self.__config.marker_padding] * (max_sequence_length - len(sequence)),batch_list))

        # Vertical padding
        if min_train_batch_size is not None:
            padding_row = [self.__config.marker_padding] * max_sequence_length

            for i in range(min_train_batch_size - len(horizontal_pad)):
                horizontal_pad.append(padding_row)

        return horizontal_pad

    def __sentence_dicts_to_sentence_matrices(self, sentence_dicts):
        print('def __sentence_dicts_to_sentence_matrices(self, sentence_dicts):')

        if os.path.isdir(self.__config.data_train_matrices) and os.listdir(os.path.join(self.__config.data_train_matrices, 'source_input'))!=[] and len(sentence_dicts) == 0:
            print('\tLoading saved sentence matrices from:',self.__config.data_train_matrices)
            for file in sorted(glob.glob(os.path.join(self.__config.data_train_matrices, 'source_input', 'sentence_source_input_examples_*.npy'))):
                sentence_source_input_examples_matrix = np.load(file)

                if sentence_source_input_examples_matrix.shape[1] != self.__config.network_max_source_sequence_length:
                    print(sentence_source_input_examples_matrix)

                self.__sentence_source_input_examples_matrices.append(sentence_source_input_examples_matrix)

                if self.__config.data_sentences_to_read_num is not None and len(self.__sentence_source_input_examples_matrices)>=self.__config.data_sentences_to_read_num:
                    break
            print('\t\tLOADED: self.__sentence_source_input_examples_matrices')

            for file in sorted(glob.glob(os.path.join(self.__config.data_train_matrices, 'target_input', 'sentence_target_input_examples_*.npy'))):
                sentence_target_input_examples_matrix = np.load(file)

                if sentence_target_input_examples_matrix.shape[1] != self.__config.network_max_target_sequence_length:
                    print(sentence_target_input_examples_matrix)

                self.__sentence_target_input_examples_matrices.append(sentence_target_input_examples_matrix)

                if self.__config.data_sentences_to_read_num is not None and len(self.__sentence_target_input_examples_matrices)>=self.__config.data_sentences_to_read_num:
                    break
            print('\t\tLOADED: self.__sentence_target_input_examples_matrices')

            for file in sorted(glob.glob(os.path.join(self.__config.data_train_matrices, 'target_output', 'sentence_target_output_examples_*.npy'))):
                sentence_target_output_examples_matrix = np.load(file)

                if sentence_target_output_examples_matrix.shape[1] != self.__config.network_max_target_sequence_length:
                    print(sentence_target_output_examples_matrix)

                self.__sentence_target_output_examples_matrices.append(sentence_target_output_examples_matrix)

                if self.__config.data_sentences_to_read_num is not None and len(self.__sentence_target_output_examples_matrices)>=self.__config.data_sentences_to_read_num:
                    break
            print('\t\tLOADED: self.__sentence_target_output_examples_matrices')

        else:
            print('\tGenerating and saving sentence matrices to empty directory: ',self.__config.data_train_matrices)
            sentence_id = 0
            for sentence_dict in sentence_dicts:
                sentence_source_input_examples, sentence_target_input_examples, sentence_target_output_examples = self.__sentence_dict_to_examples(sentence_dict)

                sentence_source_input_examples_matrix = np.matrix(sentence_source_input_examples)
                sentence_target_input_examples_matrix = np.matrix(sentence_target_input_examples)
                sentence_target_output_examples_matrix = np.matrix(sentence_target_output_examples)

                if sentence_source_input_examples_matrix.shape[1] != self.__config.network_max_source_sequence_length:
                    print('print(sentence_source_input_examples_matrix)')
                    print(sentence_source_input_examples_matrix.shape)
                    print(sentence_source_input_examples_matrix)

                if sentence_target_input_examples_matrix.shape[1] != self.__config.network_max_target_sequence_length:
                    print('print(sentence_target_input_examples_matrix)')
                    print(sentence_target_input_examples_matrix.shape)
                    print(sentence_target_input_examples_matrix)

                if sentence_target_output_examples_matrix.shape[1] != self.__config.network_max_target_sequence_length:
                    print('print(sentence_target_output_examples_matrix)')
                    print(sentence_target_output_examples_matrix.shape)
                    print(sentence_target_output_examples_matrix)

                # TODO: Check if max_*_sequence_length is too low (matrix contains list)

                np.save(os.path.join(self.__config.data_train_matrices, 'source_input', 'sentence_source_input_examples_'+str(sentence_id).zfill(10)+'.npy'), sentence_source_input_examples_matrix)
                np.save(os.path.join(self.__config.data_train_matrices, 'target_input', 'sentence_target_input_examples_'+str(sentence_id).zfill(10)+'.npy'), sentence_target_input_examples_matrix)
                np.save(os.path.join(self.__config.data_train_matrices, 'target_output', 'sentence_target_output_examples_'+str(sentence_id).zfill(10)+'.npy'), sentence_target_output_examples_matrix)

                sentence_id += 1

                self.__sentence_source_input_examples_matrices.append(sentence_source_input_examples_matrix)
                self.__sentence_target_input_examples_matrices.append(sentence_target_input_examples_matrix)
                self.__sentence_target_output_examples_matrices.append(sentence_target_output_examples_matrix)

    def get_example_matrices(self, sentence_dicts):
        print('def get_example_matrices(self, sentence_dicts):')

        if self.__sentence_source_input_examples_matrices == []:
            self.__sentence_dicts_to_sentence_matrices(sentence_dicts)


        # Splitting sentence matrices list to train, validation, test
        sentence_example_matrices_zipped = list(zip(
            self.__sentence_source_input_examples_matrices,
            self.__sentence_target_input_examples_matrices,
            self.__sentence_target_output_examples_matrices
        ))

        train_sentence_example_matrices_zipped = sentence_example_matrices_zipped[:int(len(sentence_example_matrices_zipped)*self.__config.data_train_ratio)]
        validation_sentence_example_matrices_zipped = sentence_example_matrices_zipped[
            int(len(sentence_example_matrices_zipped) * self.__config.data_train_ratio):
            int(len(sentence_example_matrices_zipped) * (self.__config.data_train_ratio + self.__config.data_validation_ratio))
        ]
        test_sentence_example_matrices_zipped = sentence_example_matrices_zipped[int(len(sentence_example_matrices_zipped) * (self.__config.data_train_ratio + self.__config.data_validation_ratio)):]

        if len(train_sentence_example_matrices_zipped) + len(validation_sentence_example_matrices_zipped) + len(test_sentence_example_matrices_zipped) != len(sentence_example_matrices_zipped):
            print('train_sentence_example_matrices_zipped', len(train_sentence_example_matrices_zipped))
            print('validation_sentence_example_matrices_zipped', len(validation_sentence_example_matrices_zipped))
            print('test_sentence_example_matrices_zipped',len(test_sentence_example_matrices_zipped))
            print('sum: ',len(train_sentence_example_matrices_zipped) + len(validation_sentence_example_matrices_zipped) + len(test_sentence_example_matrices_zipped))
            print('total: ', len(sentence_example_matrices_zipped))
            raise ValueError('SUM len of train, validation, test sentence example matrices != len of sentence_example_matrices_zipped')

        if self.__config.train_shuffle_sentences:
            print('\tShuffling train sentences...')
            shuffle(train_sentence_example_matrices_zipped)
        else:
            print('\tUsing sentences in original order.')

        train_source_input_examples = np.concatenate(tuple(sentence_example_matrices[0] for sentence_example_matrices in train_sentence_example_matrices_zipped),axis=0)
        train_target_input_examples = np.concatenate(tuple(sentence_example_matrices[1] for sentence_example_matrices in train_sentence_example_matrices_zipped),axis=0)
        train_target_output_examples = np.concatenate(tuple(sentence_example_matrices[2] for sentence_example_matrices in train_sentence_example_matrices_zipped),axis=0)

        validation_source_input_examples = np.concatenate(tuple(sentence_example_matrices[0] for sentence_example_matrices in validation_sentence_example_matrices_zipped), axis=0)
        validation_target_input_examples = np.concatenate(tuple(sentence_example_matrices[1] for sentence_example_matrices in validation_sentence_example_matrices_zipped), axis=0)
        validation_target_output_examples = np.concatenate(tuple(sentence_example_matrices[2] for sentence_example_matrices in validation_sentence_example_matrices_zipped), axis=0)

        test_source_input_examples = np.concatenate(tuple(sentence_example_matrices[0] for sentence_example_matrices in test_sentence_example_matrices_zipped), axis=0)
        test_target_input_examples = np.concatenate(tuple(sentence_example_matrices[1] for sentence_example_matrices in test_sentence_example_matrices_zipped), axis=0)
        test_target_output_examples = np.concatenate(tuple(sentence_example_matrices[2] for sentence_example_matrices in test_sentence_example_matrices_zipped), axis=0)



        if self.__config.train_batch_size > train_source_input_examples.shape[0]:
            raise ValueError("train_batch_size (" + str(self.__config.train_batch_size) + ") > #{examples}=" + str(train_source_input_examples.shape[0]))

        Dataset = namedtuple('Dataset', ['source_input_examples', 'target_input_examples', 'target_output_examples'])

        return (
            Dataset(
                source_input_examples=train_source_input_examples,
                target_input_examples=train_target_input_examples,
                target_output_examples=train_target_output_examples
            ),
            Dataset(
                source_input_examples=validation_source_input_examples,
                target_input_examples=validation_target_input_examples,
                target_output_examples=validation_target_output_examples
            ),
            Dataset(
                source_input_examples=test_source_input_examples,
                target_input_examples=test_target_input_examples,
                target_output_examples=test_target_output_examples
            )
        )

    def __sentence_dict_to_examples(self, sentence_dict):
        """Turns a sentence dict into windows and contained analyses into list format. Horizontal padding is added."""
        source_input_sequences = []
        target_input_sequences = []
        target_output_sequences = []

        analysis_id = 0
        while analysis_id+self.__config.network_window_length <= len(sentence_dict['word']):
            window_correct_analyses = sentence_dict['correct_analysis_vector'][analysis_id:analysis_id+self.__config.network_window_length]

            source_input_sequence = []
            for analysis_list in window_correct_analyses[:-1]:
                source_input_sequence.extend(analysis_list)
                source_input_sequence.append(self.__config.marker_analysis_divider)

            source_input_sequence = source_input_sequence[:-1]
            source_input_sequence.append(window_correct_analyses[-1][0])

            target_output_sequence = window_correct_analyses[-1] + [self.__config.marker_end_of_sentence]
            target_input_sequence = [self.__config.marker_go] + window_correct_analyses[-1]

            source_input_sequences.append(source_input_sequence)
            target_input_sequences.append(target_input_sequence)
            target_output_sequences.append(target_output_sequence)

            analysis_id+=1

        source_input_examples = self.pad_batch(source_input_sequences, self.__config.network_max_source_sequence_length)
        target_input_examples = self.pad_batch(target_input_sequences, self.__config.network_max_target_sequence_length)
        target_output_examples = self.pad_batch(target_output_sequences, self.__config.network_max_target_sequence_length)

        return source_input_examples, target_input_examples, target_output_examples