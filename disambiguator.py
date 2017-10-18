import tensorflow as tf
import numpy as np
import pandas as pd
import itertools

from data_processing.analyses_processor import AnalysesProcessor
from data_processing.train_data_processor import TrainDataProcessor
from model.model_inference import BuildInferenceModel
from model.model_train import BuildTrainModel


class Disambiguator(object):
    def __init__(self, model_configuration):
        self.__config = model_configuration

        # Loading train data
        self.__train_data_processor = TrainDataProcessor(model_configuration)

        self.__analyses_processor = AnalysesProcessor(model_configuration, self.__train_data_processor.vocabulary)

        # Building train model
        build_train_model = BuildTrainModel(model_configuration,
                                            self.__train_data_processor.vocabulary,
                                            self.__train_data_processor.inverse_vocabulary)

        build_train_model.create_model()

        # Building inference model
        build_inference_model = BuildInferenceModel(model_configuration,
                                                    self.__train_data_processor.inverse_vocabulary,
                                                    build_train_model)

        self.__inference_model = build_inference_model.create_model()

    def __collect_analyses_for_each_word_in_window(self, index):
        window_word_analyses = []
        for id_in_window in range(index, index + self.__config.window_length):
            word = self.__train_data_processor.corpus_dataframe.loc[id_in_window]['word']
            analyses_of_word = self.__analyses_processor.get_analyses_list_for_word(word)
            window_word_analyses.append(analyses_of_word)
            # print('\t', word, '\t', analyses_of_word)

        return window_word_analyses

    def __format_window_word_analyses(self, window_word_analyses):
        flattened_markered_combinations = []

        combinations_in_window = list(itertools.product(*window_word_analyses))

        for combination in combinations_in_window:
            combination = list(combination)

            analyses_divided = [[self.__config.marker_analysis_divider]] * (len(combination) * 2 - 1)
            analyses_divided[0::2] = combination
            flattened_list = list(itertools.chain.from_iterable(analyses_divided))

            EOS_appended = flattened_list + [self.__config.marker_end_of_sentence]

            print(len(EOS_appended), EOS_appended)

            flattened_markered_combinations.append(EOS_appended)

        return flattened_markered_combinations

    def pad_batch(self, batch_list):
        print('batch_list_size')
        print(np.matrix(batch_list).shape)
        horizontal_pad = list(map(
            lambda sequence: sequence + [self.__config.marker_padding] * (self.__config.max_source_sequence_length - len(sequence)),
            batch_list))

        print('horizontal_pad_shape')
        print(np.matrix(horizontal_pad).shape)

        padding_row = [self.__config.marker_padding] * self.__config.max_source_sequence_length

        for i in range(self.__config.inference_batch_size - len(horizontal_pad)):
            horizontal_pad.append(padding_row)

        vertical_pad = horizontal_pad

        print('vertical_pad')
        print(vertical_pad)

        return np.matrix(vertical_pad)

    def __create_analysis_window_batch_generator(self):
        self.__analyses_processor.build_analyses_dataframe_from_file(self.__config.analyses_path)

        windows_combination_heights = []
        windows_combinations = []

        corpus_df_idx = 0
        corpus_df = self.__train_data_processor.corpus_dataframe

        while corpus_df_idx <= len(corpus_df) - self.__config.window_length + 1:
            # Is last window in sentence
            if (corpus_df_idx + self.__config.window_length - 1 == len(corpus_df)) or \
                    (corpus_df.loc[corpus_df_idx + self.__config.window_length - 1]['word'] == \
                             self.__train_data_processor.inverse_vocabulary[self.__config.marker_start_of_sentence]):
                padded_sentence_batch = self.pad_batch(windows_combinations)

                print('padded_sentence_batch_shape')
                print(padded_sentence_batch.shape)

                print('padded_sentence_batch')
                print(padded_sentence_batch)

                yield windows_combination_heights, padded_sentence_batch

                windows_combination_heights = []
                windows_combinations = []

                corpus_df_idx += self.__config.window_length - 1
                continue

            # Pipeline alike processing of current word

            window_word_analyses = self.__collect_analyses_for_each_word_in_window(corpus_df_idx)

            vectorized_window_combinations = self.__format_window_word_analyses(
                window_word_analyses)

            windows_combination_heights.append(len(vectorized_window_combinations))
            windows_combinations.extend(vectorized_window_combinations)

            corpus_df_idx += 1

    def analysis_window_batches_to_logits_generator(self):
        analysis_window_batch_generator = self.__create_analysis_window_batch_generator()

        with self.__inference_model.graph.as_default():
            with tf.Session() as sess:
                saver = tf.train.Saver()

                sess.run(tf.tables_initializer())
                sess.run(tf.global_variables_initializer())

                saver.restore(sess, self.__config.train_files_save_model)

                for windows_combination_heights, padded_sentence_batch in analysis_window_batch_generator:
                    final_outputs = sess.run([self.__inference_model.final_outputs],
                                             feed_dict={
                                                 self.__inference_model.placeholders.infer_inputs: padded_sentence_batch
                                             })


                    print('rnn_output')
                    print(final_outputs[0].rnn_output)
                    # Window heights and logits of each output sequence
                    yield windows_combination_heights, map(lambda rnn_output: np.product(rnn_output.max(axis=1)), final_outputs[0].rnn_output)

