from functools import reduce

import tensorflow as tf
import numpy as np
import pandas as pd
import itertools

from operator import itemgetter

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

    def __collect_analyses_for_each_word_in_window(self, sentence_words, word_in_sentence_id, in_vector_format=True):
        window_word_analyses = []
        for id_in_window in range(word_in_sentence_id, word_in_sentence_id + self.__config.window_length):
            word = sentence_words[id_in_window]

            if in_vector_format:
                analyses_of_word = self.__analyses_processor.get_analyses_vector_list_for_word(word)
            else:
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

            #print(len(EOS_appended), EOS_appended)

            flattened_markered_combinations.append(EOS_appended)

        return flattened_markered_combinations

    def pad_batch(self, batch_list):
        horizontal_pad = list(map(
            lambda sequence: sequence + [self.__config.marker_padding] * (self.__config.max_source_sequence_length - len(sequence)),
            batch_list))

        padding_row = [self.__config.marker_padding] * self.__config.max_source_sequence_length

        for i in range(self.__config.inference_batch_size - len(horizontal_pad)):
            horizontal_pad.append(padding_row)

        batch = np.matrix(horizontal_pad)

        return batch[:self.__config.inference_batch_size]

    def __create_analysis_window_batch_generator(self, corpus_words):
        self.__analyses_processor.build_analyses_dataframe_from_file(self.__config.analyses_path)

        windows_combination_vectors_in_sentence = []
        windows_combinations_in_sentence = []

        word_in_sentence_id = 0

        while word_in_sentence_id <= len(corpus_words) - self.__config.window_length + 1:
            # Is last window in sentence
            if (word_in_sentence_id + self.__config.window_length - 1 == len(corpus_words)) or \
                    (corpus_words[word_in_sentence_id + self.__config.window_length - 1] == self.__train_data_processor.inverse_vocabulary[self.__config.marker_start_of_sentence]):
                padded_sentence_batch = self.pad_batch(windows_combination_vectors_in_sentence)

                yield windows_combinations_in_sentence, padded_sentence_batch

                windows_combination_vectors_in_sentence = []
                windows_combinations_in_sentence = []

                word_in_sentence_id += self.__config.window_length - 1
                continue

            # Pipeline alike processing of current word

            window_combinations_vector = self.__collect_analyses_for_each_word_in_window(corpus_words, word_in_sentence_id)
            window_combinations = self.__collect_analyses_for_each_word_in_window(corpus_words, word_in_sentence_id, False)

            vectorized_window_combinations = self.__format_window_word_analyses(window_combinations_vector[:self.__config.inference_batch_size])

            windows_combination_vectors_in_sentence.extend(vectorized_window_combinations)
            windows_combinations_in_sentence.append(list(itertools.product(*window_combinations))[:self.__config.inference_batch_size])

            word_in_sentence_id += 1

    def __corpus_words_to_windows_and_probabilities(self, corpus_words):
        analysis_window_batch_generator = self.__create_analysis_window_batch_generator(corpus_words)

        with self.__inference_model.graph.as_default():
            with tf.Session() as sess:
                saver = tf.train.Saver()

                sess.run(tf.tables_initializer())
                sess.run(tf.global_variables_initializer())

                saver.restore(sess, self.__config.train_files_save_model)

                for windows_combinations_in_sentence, padded_sentence_batch in analysis_window_batch_generator:
                    final_outputs = sess.run([self.__inference_model.final_outputs],
                                             feed_dict={
                                                 self.__inference_model.placeholders.infer_inputs: padded_sentence_batch
                                             })


                    # Window heights and logits of each output sequence
                    yield windows_combinations_in_sentence, map(lambda rnn_output: np.product(rnn_output.max(axis=1)), final_outputs[0].rnn_output)


    def disambiguate_text(self, text):
        # TODO: Huntoken
        sentences = text.split(". ")
        corpus_words_by_sentence = list(map(lambda sentence: sentence.split(" "), sentences))
        corpus_words = []
        for sentence_words in corpus_words_by_sentence:
            corpus_words.extend((['<SOS>']*(self.__config.window_length-1)) + sentence_words)

        sentence_id = 0
        for windows_combinations_with_probabilities_in_sentence in self.get_windows_with_probabilities_by_sentence_generator(corpus_words):
            #print('SENTENCE')
            viterbi_lists = []

            for window_combinations_with_probabilities in windows_combinations_with_probabilities_in_sentence:
                index_by_last_four_analyses = dict()
                for combination, probability in window_combinations_with_probabilities:
                    if combination[1:] in index_by_last_four_analyses.keys():
                        index_by_last_four_analyses[combination[1:]].append((combination, probability))
                    else:
                        index_by_last_four_analyses.setdefault(combination[1:], [(combination, probability)])

                reduced_groups_by_max_probability = []

                for last_four_analyses, matching_combinations_with_probability in index_by_last_four_analyses.items():
                    combination_with_max_probability_in_group = max(matching_combinations_with_probability, key=itemgetter(1))
                    reduced_groups_by_max_probability.append(combination_with_max_probability_in_group)

                viterbi_lists.append(reduced_groups_by_max_probability)

            disambiguated_combinations = []
            last_viterbi = viterbi_lists[-1]
            argmax_combination_probability_tuple = reduce(lambda max, combination_probability_tuple: combination_probability_tuple if combination_probability_tuple[1]>max[1] else max, last_viterbi)
            disambiguated_combinations.append(list(argmax_combination_probability_tuple[0]))
            for viterbi in reversed(viterbi_lists[:-1]):
                argmax_combination_probability_tuple = next(filter(lambda combination_probability_tuple: combination_probability_tuple[0][1:]==argmax_combination_probability_tuple[0][:-1], viterbi))
                disambiguated_combinations.append(list(argmax_combination_probability_tuple[0]))

            disambiguated_analyses = list(map(lambda combination: combination[-1], reversed(disambiguated_combinations)))
            #for word, correct_analysis in zip(corpus_words_by_sentence[sentence_id], disambiguated_analyses):
                #print('\t', word, '\t', correct_analysis)
            sentence_id += 1


    def get_windows_with_probabilities_by_sentence_generator(self, corpus_words):
        for windows_combinations_in_sentence, probabilities_in_sentence in self.__corpus_words_to_windows_and_probabilities(corpus_words):
            probabilities_in_sentence = list(probabilities_in_sentence)
            windows_combinations_with_probabilities = []

            windows_combination_heights_in_sentence = list(map(lambda window_combinations: len(window_combinations), windows_combinations_in_sentence))

            for window_id, window_height in enumerate(windows_combination_heights_in_sentence):
                window_heights_so_far = sum(windows_combination_heights_in_sentence[:window_id])
                window_probabilities = []
                for probability_id in range(window_heights_so_far, min(window_heights_so_far+window_height, self.__config.inference_batch_size)):
                    window_probabilities.append(probabilities_in_sentence[probability_id])

                windows_combinations_with_probabilities.append(zip([tuple(combination) for combination in windows_combinations_in_sentence[window_id]], window_probabilities))

            yield windows_combinations_with_probabilities