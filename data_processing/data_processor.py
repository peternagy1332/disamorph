import csv
import glob
import os
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
        if os.path.isdir(self.__config.train_matrices) and os.listdir(os.path.join(self.__config.train_matrices, 'source_input'))!=[] and use_saved_matrices:
            print('\tUsing previously saved train matrices. Returning no dataframes.')
            return []

        sentence_dataframes = []

        read_rows = 0
        should_stop = False
        for file in glob.glob(self.__config.train_files_corpus):
            if should_stop:
                break

            with open(file, newline='', encoding='utf8') as csvfile:
                csvreader = csv.reader(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE)

                sentence_dict = {'word': [], 'correct_analysis_vector': [], 'correct_analysis': [] }
                for i in range(self.__config.window_length - 1):
                    sentence_dict['word'].append(self.__analyses_processor.inverse_vocabulary[self.__config.marker_start_of_sentence])
                    sentence_dict['correct_analysis_vector'].append([self.__config.marker_start_of_sentence])
                    sentence_dict['correct_analysis'].append(self.__analyses_processor.inverse_vocabulary[self.__config.marker_start_of_sentence])

                csv.field_size_limit(sys.maxsize)
                for row in csvreader:
                    # End of sentence
                    if len(row) == 0:
                        sentence_dataframes.append(sentence_dict)
                        sentence_dict = {'word': [], 'correct_analysis_vector': [], 'correct_analysis': [] }
                        for i in range(self.__config.window_length - 1):
                            sentence_dict['word'].append(self.__analyses_processor.inverse_vocabulary[self.__config.marker_start_of_sentence])
                            sentence_dict['correct_analysis_vector'].append([self.__config.marker_start_of_sentence])
                            sentence_dict['correct_analysis'].append(self.__analyses_processor.inverse_vocabulary[self.__config.marker_start_of_sentence])
                        continue

                    sentence_dict['word'].append(row[0])

                    if '+?' in row[8]:
                        if self.__config.train_rebuild_vocabulary_file:
                            self.__analyses_processor.vocabulary_file.write(row[8] + os.linesep)
                        morphemes_and_tags = [row[8]]
                    else:
                        tape = self.__analyses_processor.raw_tape_to_tape(row[8])
                        morphemes_and_tags = self.__analyses_processor.tape_to_morphemes_and_tags(tape)

                    sentence_dict['correct_analysis_vector'].append(self.__analyses_processor.lookup_morphemes_and_tags_to_ids(morphemes_and_tags))
                    sentence_dict['correct_analysis'].append("".join(morphemes_and_tags))

                    read_rows+=1
                    if self.__config.rows_to_read_num is not None and read_rows>=self.__config.rows_to_read_num:
                        should_stop = True
                        break

        print('\t#{read rows}:', read_rows)
        print('\t#{total sentences}:',len(sentence_dataframes))
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
        horizontal_pad = list(map(lambda sequence: sequence + [self.__config.marker_padding] * (max_sequence_length - len(sequence)),batch_list))

        # Vertical padding
        if min_batch_size is not None:
            padding_row = [self.__config.marker_padding] * max_sequence_length

            for i in range(min_batch_size - len(horizontal_pad)):
                horizontal_pad.append(padding_row)

        return horizontal_pad

    def __train_dataframes_to_sentence_matrices(self, train_dataframes):
        print('def __train_dataframes_to_sentence_matrices(self, train_dataframes):')

        if os.path.isdir(self.__config.train_matrices) and os.listdir(os.path.join(self.__config.train_matrices, 'source_input'))!=[]:
            print('\tLoading saved sentence matrices from:',self.__config.train_matrices)
            for file in sorted(glob.glob(os.path.join(self.__config.train_matrices, 'source_input', 'sentence_source_input_examples_*.npy'))):
                sentence_source_input_examples_matrix = np.load(file)
                self.__sentence_source_input_examples_matrices.append(sentence_source_input_examples_matrix)
                if sentence_source_input_examples_matrix.shape[1] != self.__config.max_source_sequence_length:
                    print(sentence_source_input_examples_matrix)
            print('\tLOADED: self.__sentence_source_input_examples_matrices')

            for file in sorted(glob.glob(os.path.join(self.__config.train_matrices, 'target_input', 'sentence_target_input_examples_*.npy'))):
                sentence_target_input_examples_matrix = np.load(file)
                self.__sentence_target_input_examples_matrices.append(sentence_target_input_examples_matrix)
            print('\tLOADED: self.__sentence_target_input_examples_matrices')

            for file in sorted(glob.glob(os.path.join(self.__config.train_matrices, 'target_output', 'sentence_target_output_examples_*.npy'))):
                sentence_target_output_examples_matrix = np.load(file)
                self.__sentence_target_output_examples_matrices.append(sentence_target_output_examples_matrix)
            print('\tLOADED: self.__sentence_target_output_examples_matrices')

        else:
            print('\tGenerating and saving sentence matrices to empty directory: ',self.__config.train_matrices)
            sentence_id = 0
            for sentence_dataframe in train_dataframes:
                sentence_source_input_examples,\
                sentence_target_input_examples,\
                sentence_target_output_examples = self.__sentence_dataframe_to_examples(sentence_dataframe)

                sentence_source_input_examples_matrix = np.matrix(sentence_source_input_examples)
                sentence_target_input_examples_matrix = np.matrix(sentence_target_input_examples)
                sentence_target_output_examples_matrix = np.matrix(sentence_target_output_examples)

                # TODO: Check if max_*_sequence_length is too low (matrix contains list)

                np.save(os.path.join(self.__config.train_matrices, 'source_input', 'sentence_source_input_examples_'+str(sentence_id)+'.npy'), sentence_source_input_examples_matrix)
                np.save(os.path.join(self.__config.train_matrices, 'target_input', 'sentence_target_input_examples_'+str(sentence_id)+'.npy'), sentence_target_input_examples_matrix)
                np.save(os.path.join(self.__config.train_matrices, 'target_output', 'sentence_target_output_examples_'+str(sentence_id)+'.npy'), sentence_target_output_examples_matrix)

                sentence_id += 1

                self.__sentence_source_input_examples_matrices.append(sentence_source_input_examples_matrix)
                self.__sentence_target_input_examples_matrices.append(sentence_target_input_examples_matrix)
                self.__sentence_target_output_examples_matrices.append(sentence_target_output_examples_matrix)


    def get_train_examples_matrices(self, train_dataframes):
        print('def get_train_examples_matrices(self, train_dataframes):')

        if self.__sentence_source_input_examples_matrices == []:
            self.__train_dataframes_to_sentence_matrices(train_dataframes)

        if self.__config.train_shuffle_sentences:
            print('\tShuffling sentences...')
            sentence_example_matrices_zipped = list(zip(
                self.__sentence_source_input_examples_matrices,
                self.__sentence_target_input_examples_matrices,
                self.__sentence_target_output_examples_matrices
            ))

            shuffle(sentence_example_matrices_zipped)

            source_input_examples = np.concatenate(tuple(sentence_example_matrices[0] for sentence_example_matrices in sentence_example_matrices_zipped),axis=0)
            target_input_examples = np.concatenate(tuple(sentence_example_matrices[1] for sentence_example_matrices in sentence_example_matrices_zipped),axis=0)
            target_output_examples = np.concatenate(tuple(sentence_example_matrices[2] for sentence_example_matrices in sentence_example_matrices_zipped),axis=0)


        else:
            print('\tUsing sentences in original order...')
            if self.__source_input_examples is None:
                self.__source_input_examples = np.concatenate(tuple(self.__sentence_source_input_examples_matrices),axis=0)
                self.__target_input_examples = np.concatenate(tuple(self.__sentence_target_input_examples_matrices),axis=0)
                self.__target_output_examples = np.concatenate(tuple(self.__sentence_target_output_examples_matrices),axis=0)

            source_input_examples = self.__source_input_examples
            target_input_examples = self.__target_input_examples
            target_output_examples = self.__target_output_examples

        if self.__config.batch_size > source_input_examples.shape[0]:
            raise ValueError("batch_size (" + str(self.__config.batch_size) + ") > #{examples}=" + str(source_input_examples.shape[0]))

        print('source_input_examples.shape, target_input_examples.shape, target_output_examples.shape:',source_input_examples.shape, target_input_examples.shape, target_output_examples.shape)

        return source_input_examples, target_input_examples, target_output_examples

    def __sentence_dataframe_to_examples(self, sentence_dict):
        # print('def __sentence_dataframe_to_examples(self, sentence_dataframe):')
        source_input_sequences = []
        target_input_sequences = []
        target_output_sequences = []

        analysis_id = 0
        while analysis_id+self.__config.window_length <= len(sentence_dict['word']):
            window_correct_analyses = sentence_dict['correct_analysis_vector'][analysis_id:analysis_id+self.__config.window_length]

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

        source_input_examples = self.pad_batch(source_input_sequences, self.__config.max_source_sequence_length)
        target_input_examples = self.pad_batch(target_input_sequences, self.__config.max_target_sequence_length)
        target_output_examples = self.pad_batch(target_output_sequences, self.__config.max_target_sequence_length)

        return source_input_examples, target_input_examples, target_output_examples