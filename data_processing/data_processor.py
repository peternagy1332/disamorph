import csv
import glob
import os
from collections import namedtuple

import numpy as np
import itertools
import sys
from random import shuffle

import re

from utils import Colors


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

        self.__stat = {
            'longest_source_input_list': None,
            'longest_target_input_list': None
        }


    def get_sentence_dicts(self, use_saved_matrices=True):
        #print('def get_sentence_dicts(self, use_saved_matrices='+str(use_saved_matrices)+'):')
        if os.path.isdir(self.__config.data_train_matrices) and os.listdir(os.path.join(self.__config.data_train_matrices, 'source_input'))!=[] and use_saved_matrices and not self.__config.train_rebuild_vocabulary_file:
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

                sentence_dict = {'word': [], 'correct_analysis_vector': [], 'correct_analysis_tape': [], 'correct_analysis': []}
                for i in range(self.__config.network_window_length - 1):
                    sentence_dict['word'].append(self.__analyses_processor.inverse_vocabulary[self.__config.marker_start_of_sentence])
                    sentence_dict['correct_analysis_vector'].append([self.__config.marker_start_of_sentence])
                    sentence_dict['correct_analysis_tape'].append([self.__analyses_processor.inverse_vocabulary[self.__config.marker_start_of_sentence]])
                    sentence_dict['correct_analysis'].append(self.__analyses_processor.inverse_vocabulary[self.__config.marker_start_of_sentence])

                csv.field_size_limit(sys.maxsize)
                for row in csvreader:
                    # End of sentence
                    if len(row) == 0:
                        sentence_dicts.append(sentence_dict)
                        sentence_dict = {'word': [], 'correct_analysis_vector': [], 'correct_analysis_tape': [], 'correct_analysis': []}
                        for i in range(self.__config.network_window_length - 1):
                            sentence_dict['word'].append(self.__analyses_processor.inverse_vocabulary[self.__config.marker_start_of_sentence])
                            sentence_dict['correct_analysis_vector'].append([self.__config.marker_start_of_sentence])
                            sentence_dict['correct_analysis_tape'].append([self.__analyses_processor.inverse_vocabulary[self.__config.marker_start_of_sentence]])
                            sentence_dict['correct_analysis'].append(self.__analyses_processor.inverse_vocabulary[self.__config.marker_start_of_sentence])
                        continue

                    sentence_dict['word'].append(row[0])

                    # No correct analysis available
                    if '+?' == row[8][-2:]:
                        if self.__config.data_example_resolution == 'morpheme':
                            self.__analyses_processor.extend_vocabulary(row[8][:-2])
                            # Actually no tags
                            features_and_tags = [row[8][:-2]]
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

                    sentence_dict['correct_analysis_vector'].append(self.__analyses_processor.lookup_features_to_ids(features_and_tags))
                    sentence_dict['correct_analysis_tape'].append(features_and_tags)
                    sentence_dict['correct_analysis'].append("".join(features_and_tags))

                    read_rows+=1

                    if self.__config.data_sentences_to_read_num is not None and len(sentence_dicts)>=self.__config.data_sentences_to_read_num and not self.__config.train_rebuild_vocabulary_file:
                        should_stop = True
                        break

        self.__analyses_processor.write_vocabulary_file()

        print(Colors.CYAN + '\t#{total sentences}:', len(sentence_dicts))
        print(Colors.CYAN + '\t#{read rows}:', read_rows)

        return sentence_dicts

    def get_split_sentence_dicts(self, use_saved_matrices=True):
        sentence_dicts = self.get_sentence_dicts(use_saved_matrices)
        # Splitting sentence dict list to train, validation, test
        train_sentence_dicts = sentence_dicts[:int(len(sentence_dicts) * self.__config.data_train_ratio)]

        validation_sentence_dicts = sentence_dicts[
            int(len(sentence_dicts) * self.__config.data_train_ratio):
            int(len(sentence_dicts) * (self.__config.data_train_ratio + self.__config.data_validation_ratio))
        ]

        test_sentence_dicts = sentence_dicts[int(len(sentence_dicts) * (self.__config.data_train_ratio + self.__config.data_validation_ratio)):]


        print(Colors.CYAN + '\t#{train sentences}:', len(train_sentence_dicts))
        print(Colors.CYAN + '\t#{validation sentences}:', len(validation_sentence_dicts))
        print(Colors.CYAN + '\t#{test sentences}:', len(test_sentence_dicts))

        if len(train_sentence_dicts) + len(validation_sentence_dicts) + len(test_sentence_dicts) != len(sentence_dicts):
            raise ValueError('SUM len of train, validation, test sentences != len of sentences')

        return train_sentence_dicts, validation_sentence_dicts, test_sentence_dicts

    def format_window_word_analyses(self, combinations_in_window):
        flattened_markered_combinations = []

        for combination in combinations_in_window:
            combination = list(combination)

            analyses_divided = [[self.__config.marker_analysis_divider]] * (len(combination) * 2 - 1)
            analyses_divided[0::2] = combination

            flattened_list = list(itertools.chain.from_iterable(analyses_divided))


            flattened_markered_combinations.append(flattened_list)

        return flattened_markered_combinations

    def pad_batch_list(self, batch_list, max_sequence_length, min_data_batch_size = None):
        horizontal_pad = list(map(lambda sequence: sequence + [self.__config.marker_padding] * (max_sequence_length - len(sequence)),batch_list))

        # Vertical padding
        if min_data_batch_size is not None:
            padding_row = [self.__config.marker_padding] * max_sequence_length

            for i in range(min_data_batch_size - len(horizontal_pad)):
                horizontal_pad.append(padding_row)

        return horizontal_pad

    def __sentence_dicts_to_sentence_matrices(self, sentence_dicts):
        #print('def __sentence_dicts_to_sentence_matrices(self, sentence_dicts):')

        if os.path.isdir(self.__config.data_train_matrices) and os.listdir(os.path.join(self.__config.data_train_matrices, 'source_input'))!=[] and len(sentence_dicts) == 0:
            print(Colors.PINK + '\tLoading saved sentence matrices from:',self.__config.data_train_matrices)
            for file in sorted(glob.glob(os.path.join(self.__config.data_train_matrices, 'source_input', 'sentence_source_input_examples_*.npy'))):
                sentence_source_input_examples_matrix = np.load(file)

                if sentence_source_input_examples_matrix.shape[1] != self.__config.network_max_source_sequence_length:
                    raise ValueError('sentence_source_input_examples_matrix.shape[1]='+str(sentence_source_input_examples_matrix.shape[1])+'!='+str(self.__config.network_max_source_sequence_length)+\
                                     'Delete existing matrices from '+self.__config.data_train_matrices+'. Program restart will regenerate matrices with current dimensions.')

                self.__sentence_source_input_examples_matrices.append(sentence_source_input_examples_matrix)

                if self.__config.data_sentences_to_read_num is not None and len(self.__sentence_source_input_examples_matrices)>=self.__config.data_sentences_to_read_num:
                    break
            print(Colors.PINK + '\t\tLOADED: self.__sentence_source_input_examples_matrices')

            for file in sorted(glob.glob(os.path.join(self.__config.data_train_matrices, 'target_input', 'sentence_target_input_examples_*.npy'))):
                sentence_target_input_examples_matrix = np.load(file)

                if sentence_target_input_examples_matrix.shape[1] != self.__config.network_max_target_sequence_length:
                    raise ValueError('sentence_target_input_examples_matrix.shape[1]='+str(sentence_target_input_examples_matrix.shape[1])+'!='+str(self.__config.network_max_target_sequence_length)+\
                                     'Delete existing matrices from '+self.__config.data_train_matrices+'. Program restart will regenerate matrices with current dimensions.')

                self.__sentence_target_input_examples_matrices.append(sentence_target_input_examples_matrix)

                if self.__config.data_sentences_to_read_num is not None and len(self.__sentence_target_input_examples_matrices)>=self.__config.data_sentences_to_read_num:
                    break
            print(Colors.PINK + '\t\tLOADED: self.__sentence_target_input_examples_matrices')

            for file in sorted(glob.glob(os.path.join(self.__config.data_train_matrices, 'target_output', 'sentence_target_output_examples_*.npy'))):
                sentence_target_output_examples_matrix = np.load(file)

                if sentence_target_output_examples_matrix.shape[1] != self.__config.network_max_target_sequence_length:
                    raise ValueError(
                        'sentence_target_output_examples_matrix.shape[1]=' + str(sentence_target_output_examples_matrix.shape[1]) + '!=' + str(self.__config.network_max_target_sequence_length) + \
                        'Delete existing matrices from ' + self.__config.data_train_matrices + '. Program restart will regenerate matrices with current dimensions.')

                self.__sentence_target_output_examples_matrices.append(sentence_target_output_examples_matrix)

                if self.__config.data_sentences_to_read_num is not None and len(self.__sentence_target_output_examples_matrices)>=self.__config.data_sentences_to_read_num:
                    break
            print(Colors.PINK + '\t\tLOADED: self.__sentence_target_output_examples_matrices')

        else:
            print(Colors.LIGHTRED + '\tCould not load train matrices for filesystem => Starting generation.')
            #print('\tGenerating and saving sentence matrices to empty directory: ',self.__config.data_train_matrices)
            sentence_id = 0
            for sentence_dict in sentence_dicts:
                sentence_source_input_examples, sentence_target_input_examples, sentence_target_output_examples = self.sentence_dict_to_examples(sentence_dict)

                sentence_source_input_examples_matrix = np.matrix(sentence_source_input_examples)
                sentence_target_input_examples_matrix = np.matrix(sentence_target_input_examples)
                sentence_target_output_examples_matrix = np.matrix(sentence_target_output_examples)

                if sentence_source_input_examples_matrix.shape[1] != self.__config.network_max_source_sequence_length:
                    raise ValueError('sentence_source_input_examples_matrix.shape[1]='+str(sentence_source_input_examples_matrix.shape[1])+'!='+str(self.__config.network_max_source_sequence_length)+\
                                     'network_max_source_sequence_length might be too small!')

                if sentence_target_input_examples_matrix.shape[1] != self.__config.network_max_target_sequence_length:
                    raise ValueError('sentence_target_input_examples_matrix.shape[1]='+str(sentence_target_input_examples_matrix.shape[1])+'!='+str(self.__config.network_max_target_sequence_length)+\
                                     'network_max_target_sequence_length might be too small!')

                if sentence_target_output_examples_matrix.shape[1] != self.__config.network_max_target_sequence_length:
                    raise ValueError(
                        'sentence_target_output_examples_matrix.shape[1]=' + str(sentence_target_output_examples_matrix.shape[1]) + '!=' + str(self.__config.network_max_target_sequence_length) + \
                        'network_max_target_sequence_length might be too small!')


                # TODO: Check if max_*_sequence_length is too low (matrix contains list)

                if self.__config.data_save_train_matrices:
                    print(Colors.LIGHTRED + '\tSaving train matrices to: ' + self.__config.data_train_matrices)
                    np.save(os.path.join(self.__config.data_train_matrices, 'source_input', 'sentence_source_input_examples_'+str(sentence_id).zfill(10)+'.npy'), sentence_source_input_examples_matrix)
                    np.save(os.path.join(self.__config.data_train_matrices, 'target_input', 'sentence_target_input_examples_'+str(sentence_id).zfill(10)+'.npy'), sentence_target_input_examples_matrix)
                    np.save(os.path.join(self.__config.data_train_matrices, 'target_output', 'sentence_target_output_examples_'+str(sentence_id).zfill(10)+'.npy'), sentence_target_output_examples_matrix)

                sentence_id += 1

                self.__sentence_source_input_examples_matrices.append(sentence_source_input_examples_matrix)
                self.__sentence_target_input_examples_matrices.append(sentence_target_input_examples_matrix)
                self.__sentence_target_output_examples_matrices.append(sentence_target_output_examples_matrix)

        if self.__stat['longest_source_input_list'] is not None:
            print(Colors.CYAN + '\t#{longest source input list}:', self.__stat['longest_source_input_list'])
            print(Colors.CYAN + '\t#{longest target input list}:', self.__stat['longest_target_input_list'])

    def get_example_matrices(self, sentence_dicts):
        #print('def get_example_matrices(self, sentence_dicts):')

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
            print(Colors.LIGHTBLUE + '\tShuffling train sentences...')
            shuffle(train_sentence_example_matrices_zipped)
        else:
            print(Colors.LIGHTBLUE + '\tUsing sentences in original order.')

        # with open('toy_train_source_sequences.txt','w',encoding='utf8') as f:
        #     for si, ti, to in train_sentence_example_matrices_zipped:
        #         for sirow, tirow, torow in zip(si.tolist(), ti.tolist(), to.tolist()):
        #             # f.write("".join(self.analyses_processor.lookup_ids_to_features(sirow)).replace("<PAD>", "") +'\t' +
        #             #         "".join(self.analyses_processor.lookup_ids_to_features(tirow)).replace("<PAD>", "") + '\t' +
        #             #         "".join(self.analyses_processor.lookup_ids_to_features(torow)).replace("<PAD>", "") + os.linesep)
        #             f.write("".join(self.analyses_processor.lookup_ids_to_features(sirow)).replace("<PAD>", "") + os.linesep)
        #         f.write(os.linesep)


        train_source_input_examples = np.concatenate(tuple(sentence_example_matrices[0] for sentence_example_matrices in train_sentence_example_matrices_zipped),axis=0)
        train_target_input_examples = np.concatenate(tuple(sentence_example_matrices[1] for sentence_example_matrices in train_sentence_example_matrices_zipped),axis=0)
        train_target_output_examples = np.concatenate(tuple(sentence_example_matrices[2] for sentence_example_matrices in train_sentence_example_matrices_zipped),axis=0)

        validation_source_input_examples = np.concatenate(tuple(sentence_example_matrices[0] for sentence_example_matrices in validation_sentence_example_matrices_zipped), axis=0)
        validation_target_input_examples = np.concatenate(tuple(sentence_example_matrices[1] for sentence_example_matrices in validation_sentence_example_matrices_zipped), axis=0)
        validation_target_output_examples = np.concatenate(tuple(sentence_example_matrices[2] for sentence_example_matrices in validation_sentence_example_matrices_zipped), axis=0)

        test_source_input_examples = np.concatenate(tuple(sentence_example_matrices[0] for sentence_example_matrices in test_sentence_example_matrices_zipped), axis=0)
        test_target_input_examples = np.concatenate(tuple(sentence_example_matrices[1] for sentence_example_matrices in test_sentence_example_matrices_zipped), axis=0)
        test_target_output_examples = np.concatenate(tuple(sentence_example_matrices[2] for sentence_example_matrices in test_sentence_example_matrices_zipped), axis=0)

        if self.__config.data_batch_size > train_source_input_examples.shape[0]:
            raise ValueError("data_batch_size (" + str(self.__config.data_batch_size) + ") > #{examples}=" + str(train_source_input_examples.shape[0]))

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

    def sentence_dict_to_examples(self, sentence_dict):
        """Turns a sentence dict into windows and contained analyses into list format. Horizontal padding is added."""
        source_input_sequences = []
        target_input_sequences = []
        target_output_sequences = []

        analysis_id = 0
        while analysis_id+self.__config.network_window_length <= len(sentence_dict['word']):
            window_correct_analysis_vectors = sentence_dict['correct_analysis_vector'][analysis_id:analysis_id+self.__config.network_window_length]
            window_correct_analysis_tapes = sentence_dict['correct_analysis_tape'][analysis_id:analysis_id + self.__config.network_window_length]

            ##
            # source_input_sequence = []
            # for analysis_list in window_correct_analysis[:-1]:
            #     source_input_sequence.extend(analysis_list)
            #     source_input_sequence.append(' ')
            #
            # feature_or_tag_buffer = []
            # for feature_or_tag in window_correct_analysis[-1].split(' '):
            #     if re.match(r'\[[^]]*\]', feature_or_tag, re.UNICODE):
            #         break
            #
            #     feature_or_tag_buffer.append(feature_or_tag)
            #
            # source_input_sequence.extend(" ".join(feature_or_tag_buffer))
            #
            # for i in source_input_sequence:
            #     print(i,end='')
            # print(end='\t')
            #
            # print(window_correct_analysis[-1])
            ##

            # First N-1 analyses
            source_input_sequence = []
            for analysis_list in window_correct_analysis_vectors[:-1]:
                source_input_sequence.extend(analysis_list)
                source_input_sequence.append(self.__config.marker_analysis_divider)

            # In addition, giving some future information about the Nth analysis
            extra_info = self.__analyses_processor.get_extra_info_vector(window_correct_analysis_tapes[-1])
            source_input_sequence.extend(extra_info)
            # if self.__config.data_example_resolution == 'character':
            #     feature_or_tag_buffer = []
            #     for feature_or_tag in window_correct_analysis_tapes[-1]:
            #         if re.match(r'\[[^]]*\]', feature_or_tag, re.UNICODE):
            #             break
            #
            #         feature_or_tag_buffer.append(self.analyses_processor.vocabulary.get(feature_or_tag, self.__config.marker_unknown))
            #
            #     source_input_sequence.extend(feature_or_tag_buffer)
            # elif self.__config.data_example_resolution == 'morpheme':
            #     source_input_sequence.append(window_correct_analysis_vectors[-1][0])

            target_output_sequence = window_correct_analysis_vectors[-1] + [self.__config.marker_end_of_sentence]
            target_input_sequence = [self.__config.marker_go] + window_correct_analysis_vectors[-1]

            # print("".join(self.analyses_processor.lookup_ids_to_features(source_input_sequence)),'\t',\
            #       "".join(self.analyses_processor.lookup_ids_to_features(target_output_sequence)))

            # Stat
            if self.__stat['longest_source_input_list'] is None:
                self.__stat['longest_source_input_list'] = len(source_input_sequence)
            else:
                if len(source_input_sequence) > self.__stat['longest_source_input_list']:
                    self.__stat['longest_source_input_list'] = len(source_input_sequence)

            if self.__stat['longest_target_input_list'] is None:
                self.__stat['longest_target_input_list'] = len(target_input_sequence)
            else:
                if len(target_input_sequence) > self.__stat['longest_target_input_list']:
                    self.__stat['longest_target_input_list'] = len(target_input_sequence)

            source_input_sequences.append(source_input_sequence)
            target_input_sequences.append(target_input_sequence)
            target_output_sequences.append(target_output_sequence)

            analysis_id+=1


        source_input_examples = self.pad_batch_list(source_input_sequences, self.__config.network_max_source_sequence_length)
        target_input_examples = self.pad_batch_list(target_input_sequences, self.__config.network_max_target_sequence_length)
        target_output_examples = self.pad_batch_list(target_output_sequences, self.__config.network_max_target_sequence_length)

        return source_input_examples, target_input_examples, target_output_examples