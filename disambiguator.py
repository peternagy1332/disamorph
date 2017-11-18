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


    def __collect_analyses_for_source_words_in_window(self, sentence_words, word_in_sentence_id, in_vector_format=True):
        window_word_analyses = []
        for id_in_window in range(word_in_sentence_id, word_in_sentence_id + self.__config.network_window_length):
            word = sentence_words[id_in_window]

            if in_vector_format:
                analyses_of_word = self.__analyses_processor.get_lookedup_feature_list_analyses_for_word(word)
            else:
                analyses_of_word = self.__analyses_processor.get_analyses_list_for_word(word)

            window_word_analyses.append(analyses_of_word)

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

            window_combinations_vector = self.__collect_analyses_for_source_words_in_window(corpus_words, word_in_sentence_id)
            window_analyses = self.__collect_analyses_for_source_words_in_window(corpus_words, word_in_sentence_id, False)

            last_word = corpus_words[word_in_sentence_id+self.__config.network_window_length-1]
            window_combinations_vector[-1] = self.__analyses_processor.get_all_extra_info_vectors_for_word(last_word)

            # TODO: check truncation (window_combinations_vector[:self.__config.inference_batch_size])
            combinations_in_window = list(itertools.product(*window_combinations_vector))

            vectorized_window_combinations = self.__data_processor.format_window_word_analyses(combinations_in_window)

            window_combinations = list(itertools.product(*window_analyses))

            windows_combination_vectors_in_sentence.extend(vectorized_window_combinations)
            windows_combinations_in_sentence.append(window_combinations)

            word_in_sentence_id += 1

    def __corpus_words_to_windows_and_probabilities(self, corpus_words, return_network_output):
        with self.__inference_model.graph.as_default():
            saver = tf.train.Saver()

            saver.restore(self.__inference_session, os.path.join(self.__config.model_directory, 'validation_checkpoints', self.__config.model_name))

            for windows_combinations_in_sentence, padded_sentence_batch in self.__create_analysis_window_batch_generator(corpus_words):
                # If the all the combinations was less or equal then inference_batch_size
                if len(padded_sentence_batch) == self.__config.inference_batch_size:
                    padded_sentence_batch = np.matrix(padded_sentence_batch)
                    logits = self.__inference_session.run(
                        [self.__inference_model.logits],
                        feed_dict={
                            self.__inference_model.placeholders.infer_inputs: padded_sentence_batch
                        }
                    )

                    probabilities = np.sum(np.log(np.max(logits[0], axis=2)), axis=1).tolist()

                    if return_network_output:
                        output_sequences = self.__inference_session.run(
                            [self.__inference_model.output_sequences],
                            feed_dict={
                                self.__inference_model.placeholders.infer_inputs: padded_sentence_batch
                            }
                        )
                    else:
                        output_sequences = None

                    yield windows_combinations_in_sentence, probabilities, output_sequences[0]


                else:
                    probabilities = []
                    i = 0
                    if return_network_output:
                        output_sequences = []
                    else:
                        output_sequences = None

                    while i < len(padded_sentence_batch):
                        if i+self.__config.inference_batch_size <= len(padded_sentence_batch):
                            padded_sentence_part_batch = np.matrix(padded_sentence_batch[i:i+self.__config.inference_batch_size])
                        else:
                            padded_sentence_part_batch = np.matrix(
                                self.__data_processor.pad_batch(
                                    padded_sentence_batch[i:],
                                    self.__config.network_max_source_sequence_length,
                                    self.__config.inference_batch_size
                                )
                            )
                        logits = self.__inference_session.run(
                            [self.__inference_model.logits],
                            feed_dict={
                                self.__inference_model.placeholders.infer_inputs: padded_sentence_part_batch
                            }
                        )

                        inference_batch_probabilities = np.sum(np.log(np.max(logits[0], axis=2)), axis=1).tolist()
                        probabilities.extend(inference_batch_probabilities)

                        if return_network_output:
                            r_output_sequences = self.__inference_session.run(
                                [self.__inference_model.output_sequences],
                                feed_dict={
                                    self.__inference_model.placeholders.infer_inputs: padded_sentence_part_batch
                                }
                            )

                            horizontal_padding = np.zeros(shape=(self.__config.inference_batch_size, self.__config.network_max_target_sequence_length-r_output_sequences[0].shape[1]))
                            padded_output_sequences = np.concatenate((r_output_sequences[0], horizontal_padding), axis=1)
                            output_sequences.append(padded_output_sequences)


                        i+=self.__config.inference_batch_size

                    if return_network_output:
                        output_sequences = np.concatenate(tuple(partial_output_sequence for partial_output_sequence in output_sequences), axis=0)

                    yield windows_combinations_in_sentence, probabilities, output_sequences


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

    def disambiguated_analyses_by_sentence_generator(self, corpus_words, return_output_sequences=False):
        """Viterbi. Doesn't keep <SOS>. Yields only disambiguated analyses in a list by sentence."""
        for windows_combinations_with_probabilities_in_sentence, output_sequences in self.get_windows_with_probabilities_by_sentence_generator(corpus_words, return_output_sequences):
            viterbi_lists = []

            for window_combinations_with_probabilities in windows_combinations_with_probabilities_in_sentence:
                empty_window = True # If the sentence was too large to fit in the inference input matrix
                index_by_last_four_analyses = dict()
                for combination, probability in window_combinations_with_probabilities:
                    empty_window = False

                    if combination[1:] in index_by_last_four_analyses.keys():
                        index_by_last_four_analyses[combination[1:]].append((combination, probability))
                    else:
                        index_by_last_four_analyses.setdefault(combination[1:], [(combination, probability)])

                if not empty_window:
                    reduced_groups_by_max_probability = []

                    for last_four_analyses, matching_combinations_with_probability in index_by_last_four_analyses.items():
                        combination_with_max_probability_in_group = max(matching_combinations_with_probability, key=itemgetter(1))
                        reduced_groups_by_max_probability.append(combination_with_max_probability_in_group)

                    viterbi_lists.append(reduced_groups_by_max_probability)

            disambiguated_combinations = []

            last_viterbi = viterbi_lists[-1]

            argmax_combination_probability_tuple = reduce(lambda max, combination_probability_tuple: combination_probability_tuple if combination_probability_tuple[1]>max[1] else max, last_viterbi, (['???'],-100))
            disambiguated_combinations.append(list(argmax_combination_probability_tuple[0]))
            for viterbi in reversed(viterbi_lists[:-1]):
                argmax_combination_probability_tuple = next(filter(lambda combination_probability_tuple: combination_probability_tuple[0][1:]==argmax_combination_probability_tuple[0][:-1], viterbi))
                disambiguated_combinations.append(list(argmax_combination_probability_tuple[0]))

            disambiguated_analyses = list(map(lambda combination: combination[-1], reversed(disambiguated_combinations)))
            if return_output_sequences:
                yield disambiguated_analyses, output_sequences
            else:
                yield disambiguated_analyses

    def get_windows_with_probabilities_by_sentence_generator(self, corpus_words, return_output_sequences):
        for windows_combinations_in_sentence, probabilities_in_sentence, output_sequences in self.__corpus_words_to_windows_and_probabilities(corpus_words, return_output_sequences):
            windows_combinations_with_probabilities = []

            windows_combination_heights_in_sentence = list(map(lambda window_combinations: len(window_combinations), windows_combinations_in_sentence))

            for window_id, window_height in enumerate(windows_combination_heights_in_sentence):
                window_heights_so_far = sum(windows_combination_heights_in_sentence[:window_id])
                window_probabilities = []
                for probability_id in range(window_heights_so_far, window_heights_so_far+window_height):
                    window_probabilities.append(probabilities_in_sentence[probability_id])

                window_combinations_in_sentence = [tuple(combination) for combination in windows_combinations_in_sentence[window_id]]

                windows_combinations_with_probabilities.append(zip(window_combinations_in_sentence, window_probabilities))

            yield windows_combinations_with_probabilities, output_sequences

    def evaluate_model(self, sentence_dicts, printAnalyses = False):
        disambiguation_accuracies = []
        print('\t#{sentences}: ', len(sentence_dicts))
        for sentence_id, sentence_dict in enumerate(sentence_dicts):
            if printAnalyses:
                print('\t\tSentence ', (sentence_id+1),'/', len(sentence_dicts))
            words_to_disambiguate = sentence_dict['word']  # Including <SOS>

            disambiguated_sentence, output_sequences = next(self.disambiguated_analyses_by_sentence_generator(words_to_disambiguate, True))
            correct_analyses = sentence_dict['correct_analysis'][self.__config.network_window_length - 1:]

            source_input_examples, target_input_examples, target_output_examples = self.__data_processor.sentence_dict_to_examples(sentence_dict)
            target_output_examples = np.matrix(target_output_examples, dtype=np.int32)

            output_sequences = output_sequences[:target_output_examples.shape[0]]

            target_sequence_lengths = np.count_nonzero(target_output_examples, axis=1)

            output_sequence_masks = []
            for target_sequence_length in target_sequence_lengths.tolist():
                mask = [1]*target_sequence_length[0] + [0]*(self.__config.network_max_target_sequence_length-target_sequence_length[0])
                output_sequence_masks.append(mask)

            target_weights = np.matrix(output_sequence_masks, dtype=np.int32)

            correct_elements = np.equal(target_output_examples, np.multiply(output_sequences,target_weights)).astype(np.float32)

            network_output_accuracy = np.mean(correct_elements)*100

            matching_analyses = 0
            if printAnalyses:
                print('\t\t\t%-50s%-s' % ("Correct analyses", "Disambiguated analyses"))
                print('\t\t\t'+ ('-' * 100))
            for i in range(len(disambiguated_sentence)):
                if printAnalyses:
                    print('\t\t\t%-50s%-s' % (correct_analyses[i], '|  ' + disambiguated_sentence[i]))
                if disambiguated_sentence[i] == correct_analyses[i]:
                    matching_analyses += 1

            disambiguation_accuracy = 100 * matching_analyses / len(disambiguated_sentence)
            disambiguation_accuracies.append(disambiguation_accuracy)
            if printAnalyses:
                print('\t\t\t' + ('-' * 100))
                print('\t\t\t>> Disambiguation accuracy: %.2f\n' % (disambiguation_accuracy))
            else:
                print('\t\tSentence: %8d/%-8d Disambiguation accuracy: %-6.2f Network output accuracy: %.2f' % (sentence_id + 1, len(sentence_dicts), disambiguation_accuracy, network_output_accuracy))

            if (sentence_id+1) % 5 == 0:
                print('\tAverage disambiguation accuracy: %-6.2f Network output accuracy: %.2f' % (sum(disambiguation_accuracies)/len(disambiguation_accuracies), network_output_accuracy))


        print('>>>> Disambiguation accuracies\t-\tmin: %-12.2f max: %-12.2f avg: %-12.2f\n' % (min(disambiguation_accuracies), max(disambiguation_accuracies), sum(disambiguation_accuracies)/len(disambiguation_accuracies)))
        print('='*100)