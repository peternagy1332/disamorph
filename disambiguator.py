import subprocess
from functools import reduce
from tempfile import NamedTemporaryFile
import xml.etree.ElementTree as ET

import tensorflow as tf
import numpy as np
import itertools

from operator import itemgetter
import os

from data_processing.analyses_processor import AnalysesProcessor
from data_processing.data_processor import DataProcessor
from model.model_inference import BuildInferenceModel
from model.model_train import BuildTrainModel


class Disambiguator(object):
    def __init__(self, model_configuration, analyses_processor = None):
        self.__config = model_configuration

        if analyses_processor is None:
            self.__analyses_processor = AnalysesProcessor(model_configuration)
        else:
            self.__analyses_processor = analyses_processor

        # Loading train data
        self.__data_processor = DataProcessor(model_configuration, analyses_processor)

        # Building train model
        build_train_model = BuildTrainModel(model_configuration, self.__analyses_processor.vocabulary, self.__analyses_processor.inverse_vocabulary)

        build_train_model.create_model()

        # Building inference model
        build_inference_model = BuildInferenceModel(model_configuration, self.__analyses_processor.inverse_vocabulary, build_train_model)

        self.__inference_model = build_inference_model.create_model()

        self.__inference_session = tf.Session(graph=self.__inference_model.graph)


    def __collect_analyses_for_each_word_in_window(self, sentence_words, word_in_sentence_id, in_vector_format=True):
        window_word_analyses = []
        for id_in_window in range(word_in_sentence_id, word_in_sentence_id + self.__config.network_window_length):
            word = sentence_words[id_in_window]

            if in_vector_format:
                analyses_of_word = self.__analyses_processor.get_analyses_vector_list_for_word(word)
            else:
                analyses_of_word = self.__analyses_processor.get_analyses_list_for_word(word)

            window_word_analyses.append(analyses_of_word)

            # print('\t', word, '\t', analyses_of_word)

        return window_word_analyses


    def __create_analysis_window_batch_generator(self, corpus_words):
        windows_combination_vectors_in_sentence = []
        windows_combinations_in_sentence = []

        word_in_sentence_id = 0

        while word_in_sentence_id <= len(corpus_words) - self.__config.network_window_length + 1:
            # Is last window in sentence
            if (word_in_sentence_id + self.__config.network_window_length - 1 == len(corpus_words)) or \
                    (corpus_words[word_in_sentence_id + self.__config.network_window_length - 1] == self.__analyses_processor.inverse_vocabulary[self.__config.marker_start_of_sentence]):

                padded_sentence_batch = self.__data_processor.pad_batch(
                    windows_combination_vectors_in_sentence,
                    self.__config.network_max_source_sequence_length,
                    self.__config.inference_batch_size
                )

                yield windows_combinations_in_sentence, padded_sentence_batch

                windows_combination_vectors_in_sentence = []
                windows_combinations_in_sentence = []

                word_in_sentence_id += self.__config.network_window_length - 1
                continue

            # Pipeline alike processing of current word

            window_combinations_vector = self.__collect_analyses_for_each_word_in_window(corpus_words, word_in_sentence_id)
            window_analyses = self.__collect_analyses_for_each_word_in_window(corpus_words, word_in_sentence_id, False)

            combinations_in_window = list(itertools.product(*window_combinations_vector[:self.__config.inference_batch_size]))


            vectorized_window_combinations = self.__data_processor.format_window_word_analyses(combinations_in_window)

            window_combinations = list(itertools.product(*window_analyses))

            windows_combination_vectors_in_sentence.extend(vectorized_window_combinations)
            windows_combinations_in_sentence.append(window_combinations)

            word_in_sentence_id += 1

    def __corpus_words_to_windows_and_probabilities(self, corpus_words):
        with self.__inference_model.graph.as_default():
            saver = tf.train.Saver()

            saver.restore(self.__inference_session, os.path.join(self.__config.model_directory, self.__config.model_name))

            for windows_combinations_in_sentence, padded_sentence_batch in self.__create_analysis_window_batch_generator(corpus_words):
                # If the all the combinations was less or equal then inference_batch_size
                if len(padded_sentence_batch) == self.__config.inference_batch_size:
                    padded_sentence_batch = np.matrix(padded_sentence_batch)
                    final_outputs = self.__inference_session.run(
                        [self.__inference_model.final_outputs],
                        feed_dict={
                            self.__inference_model.placeholders.infer_inputs: padded_sentence_batch
                        }
                    )

                    probabilities = np.sum(np.log(np.max(final_outputs[0].rnn_output, axis=2)),axis=1).tolist()


                    # Combinations and probabilities in window
                    #yield windows_combinations_in_sentence, list(map(lambda rnn_output: np.sum(rnn_output.max(axis=1)), final_outputs[0].rnn_output)) # log
                    #yield windows_combinations_in_sentence, list(map(lambda rnn_output: np.product(rnn_output.max(axis=1)), final_outputs[0].rnn_output)) # probability
                    print('yield')
                    yield windows_combinations_in_sentence, probabilities

                else:
                    probabilities = []
                    i=0
                    # print('len(padded_sentence_batch)', len(padded_sentence_batch))
                    while i < len(padded_sentence_batch):
                        # print('while i < len(padded_sentence_batch):', i, i+self.__config.inference_batch_size)
                        if i+self.__config.inference_batch_size <= len(padded_sentence_batch):
                            # print('\tunderindexing')
                            padded_sentence_part_batch = np.matrix(padded_sentence_batch[i:i+self.__config.inference_batch_size])
                        else:
                            # print('\toverindexing')
                            padded_sentence_part_batch = np.matrix(
                                self.__data_processor.pad_batch(
                                    padded_sentence_batch[i:],
                                    self.__config.network_max_source_sequence_length,
                                    self.__config.inference_batch_size
                                )
                            )
                        final_outputs = self.__inference_session.run(
                            [self.__inference_model.final_outputs],
                            feed_dict={
                                self.__inference_model.placeholders.infer_inputs: padded_sentence_part_batch
                            }
                        )


                        #inference_batch_probabilities = list(map(lambda rnn_output: np.sum(rnn_output.max(axis=1)), final_outputs[0].rnn_output))
                        # print('\tinference_batch_probabilities', len(inference_batch_probabilities))
                        inference_batch_probabilities = np.sum(np.log(np.max(final_outputs[0].rnn_output, axis=2)), axis=1).tolist()
                        probabilities.extend(inference_batch_probabilities)

                        i+=self.__config.inference_batch_size

                    # print('NAGY OSSZESITO', sum(len(w) for w in windows_combinations_in_sentence), len(probabilities))

                    # Combinations and probabilities in window
                    print('yield')
                    yield windows_combinations_in_sentence, probabilities

    def corpus_to_tokenized_sentences(self, corpus):
        corpus_temp_file = NamedTemporaryFile(delete=False)
        corpus_temp_file.write(corpus.encode())
        corpus_temp_file.close()

        hfst_pipe = subprocess.Popen(
            "quntoken " + corpus_temp_file.name,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True
        )

        raw_output, err = hfst_pipe.communicate(corpus.encode())
        raw_decoded_output = raw_output.decode('utf-8')

        if len(err)>0:
            print(err.decode())

        root = ET.fromstring("<root>" + raw_decoded_output + "</root>")

        tokenized_sentences = []
        for sentence in root:
            sentence_tokens = []
            for token in sentence:
                if token.text != ' ':
                    sentence_tokens.append(token.text)

            if len(sentence_tokens) > 0:
                tokenized_sentences.append(sentence_tokens)

        return tokenized_sentences

    def disambiguate_tokenized_sentences(self, tokenized_sentences):
        flattened_corpus_words_with_SOS_before_sentences = []
        for tokenized_sentence in tokenized_sentences:
            flattened_corpus_words_with_SOS_before_sentences += [self.__analyses_processor.inverse_vocabulary[self.__config.marker_start_of_sentence]]*(self.__config.network_window_length-1) +tokenized_sentence

        disambiguated_analyses_for_sentences = []
        for disambiguated_analyses in self.disambiguated_analyses_by_sentence_generator(flattened_corpus_words_with_SOS_before_sentences):
            disambiguated_analyses_for_sentences.append(disambiguated_analyses)

        return disambiguated_analyses_for_sentences

    def disambiguated_analyses_by_sentence_generator(self, corpus_words):
        """Viterbi. Doesn't keep <SOS>. Yields only disambiguated analyses in a list by sentence."""
        #sentence_id = 0
        for windows_combinations_with_probabilities_in_sentence in self.get_windows_with_probabilities_by_sentence_generator(corpus_words):
            #print('SENTENCE')
            viterbi_lists = []

            for window_combinations_with_probabilities in windows_combinations_with_probabilities_in_sentence:
                empty_window = True # If the sentence was too large to fit in the inference input matrix
                index_by_last_four_analyses = dict()
                #print('\twindow_combinations_with_probabilities')
                for combination, probability in window_combinations_with_probabilities:
                    empty_window = False
                    #print('\t\t',combination)
                    if combination[1:] in index_by_last_four_analyses.keys():
                        index_by_last_four_analyses[combination[1:]].append((combination, probability))
                    else:
                        index_by_last_four_analyses.setdefault(combination[1:], [(combination, probability)])

                if not empty_window:
                    reduced_groups_by_max_probability = []

                    #print('\tindex_by_last_four_analyses', len(index_by_last_four_analyses.keys()))
                    for last_four_analyses, matching_combinations_with_probability in index_by_last_four_analyses.items():
                        #print(last_four_analyses, len(matching_combinations_with_probability))
                        combination_with_max_probability_in_group = max(matching_combinations_with_probability, key=itemgetter(1))
                        reduced_groups_by_max_probability.append(combination_with_max_probability_in_group)

                    #print('\treduced_groups_by_max_probability', len(reduced_groups_by_max_probability))

                    viterbi_lists.append(reduced_groups_by_max_probability)

            disambiguated_combinations = []

            last_viterbi = viterbi_lists[-1]

            argmax_combination_probability_tuple = reduce(lambda max, combination_probability_tuple: combination_probability_tuple if combination_probability_tuple[1]>max[1] else max, last_viterbi, (['???'],-100))
            disambiguated_combinations.append(list(argmax_combination_probability_tuple[0]))
            for viterbi in reversed(viterbi_lists[:-1]):
                #print(argmax_combination_probability_tuple)
                argmax_combination_probability_tuple = next(filter(lambda combination_probability_tuple: combination_probability_tuple[0][1:]==argmax_combination_probability_tuple[0][:-1], viterbi))
                disambiguated_combinations.append(list(argmax_combination_probability_tuple[0]))

            #for i in disambiguated_combinations:
            #    print(i)

            disambiguated_analyses = list(map(lambda combination: combination[-1], reversed(disambiguated_combinations)))
            yield disambiguated_analyses

    def get_windows_with_probabilities_by_sentence_generator(self, corpus_words):
        for windows_combinations_in_sentence, probabilities_in_sentence in self.__corpus_words_to_windows_and_probabilities(corpus_words):
            windows_combinations_with_probabilities = []

            windows_combination_heights_in_sentence = list(map(lambda window_combinations: len(window_combinations), windows_combinations_in_sentence))

            for window_id, window_height in enumerate(windows_combination_heights_in_sentence):
                window_heights_so_far = sum(windows_combination_heights_in_sentence[:window_id])
                window_probabilities = []
                for probability_id in range(window_heights_so_far, window_heights_so_far+window_height):
                    window_probabilities.append(probabilities_in_sentence[probability_id])

                window_combinations_in_sentence = [tuple(combination) for combination in windows_combinations_in_sentence[window_id]]

                #print(len(window_combinations_in_sentence), windows_combination_heights_in_sentence[window_id], len(window_probabilities))

                windows_combinations_with_probabilities.append(zip(window_combinations_in_sentence, window_probabilities))

            yield windows_combinations_with_probabilities

    def evaluate_model(self, sentence_dataframes, printAnalyses = False):
        accuracies = []
        print('\t#{sentences}: ', len(sentence_dataframes))
        for test_sentence_id, test_sentence_dict in enumerate(sentence_dataframes):
            if printAnalyses:
                print('\t\tSentence ',(test_sentence_id+1),'/',len(sentence_dataframes))
            words_to_disambiguate = test_sentence_dict['word']  # Including <SOS>

            disambiguated_sentence = next(self.disambiguated_analyses_by_sentence_generator(words_to_disambiguate))
            correct_analyses = test_sentence_dict['correct_analysis'][self.__config.network_window_length - 1:]

            matching_analyses = 0
            if printAnalyses:
                print('\t\t\t%-50s%-s' % ("Correct analyses", "Disambiguated analyses"))
                print('\t\t\t'+ ('-' * 100))
            for i in range(len(disambiguated_sentence)):
                if printAnalyses:
                    print('\t\t\t%-50s%-s' % (correct_analyses[i], '|  ' + disambiguated_sentence[i]))
                if disambiguated_sentence[i] == correct_analyses[i]:
                    matching_analyses += 1

            accuracy = 100 * matching_analyses / len(disambiguated_sentence)
            accuracies.append(accuracy)
            if printAnalyses:
                print('\t\t\t' + ('-' * 100))
                print('\t\t\t>> Disambiguation accuracy: %.2f\n' % (accuracy))
            else:
                print('\t\tSentence: %8d/%-8d Disambiguation accuracy: %.2f' % (test_sentence_id+1, len(sentence_dataframes), accuracy))

            if (test_sentence_id+1) % 5 == 0:
                print('\tAverage disambiguation accuracy so far: %.2f' % (sum(accuracies)/len(accuracies)))


        print('>>>> Disambiguation accuracies\t-\tmin: %-12.2f max: %-12.2f avg: %-12.2f\n' % (min(accuracies), max(accuracies), sum(accuracies)/len(accuracies)))
        print('='*100)