import tensorflow as tf
import numpy as np
import itertools
import os
import subprocess
import xml.etree.ElementTree as ET
#import matplotlib.pyplot as plt
from functools import reduce
from operator import itemgetter
from tempfile import NamedTemporaryFile
from disamorph.data_processing.analyses_processor import AnalysesProcessor
from disamorph.data_processing.data_processor import DataProcessor
from disamorph.model.model_inference import BuildInferenceModel


class Disambiguator(object):
    def __init__(self, model_configuration, analyses_processor = None):
        self.__config = model_configuration

        if analyses_processor is None:
            self.analyses_processor = AnalysesProcessor(model_configuration)
        else:
            self.analyses_processor = analyses_processor

        # Loading train data
        self.__data_processor = DataProcessor(model_configuration, analyses_processor)

        # Building inference model
        build_inference_model = BuildInferenceModel(model_configuration, self.analyses_processor)

        self.__inference_model = build_inference_model.create_model()

        self.__correct_combinations = None

        self.__inference_session = tf.Session(graph=self.__inference_model.graph)

        # Restoring network weights
        with self.__inference_model.graph.as_default():
            inference_saver = tf.train.Saver()

            if self.__config.use_train_model:
                #inference_saver.restore(self.__inference_session, os.path.join(self.__config.model_directory, 'train_checkpoints', self.__config.model_name))
                latest_checkpoint = tf.train.latest_checkpoint(os.path.join(self.__config.model_directory, 'train_checkpoints'))
            else:
                #inference_saver.restore(self.__inference_session, os.path.join(self.__config.model_directory, 'validation_checkpoints', self.__config.model_name))
                latest_checkpoint = tf.train.latest_checkpoint(os.path.join(self.__config.model_directory, 'validation_checkpoints'))

            inference_saver.restore(self.__inference_session, latest_checkpoint)

    def __collect_analyses_for_source_words_in_window(self, sentence_words, word_in_sentence_id, in_vector_format=True):
        window_word_analyses = []
        for id_in_window in range(word_in_sentence_id, word_in_sentence_id + self.__config.network_window_length):
            word = sentence_words[id_in_window]

            if in_vector_format:
                analyses_of_word = self.analyses_processor.get_lookedup_feature_list_analyses_for_word(word)
            else:
                analyses_of_word = self.analyses_processor.get_analyses_list_for_word(word)

            window_word_analyses.append(analyses_of_word)

        return window_word_analyses

    def __create_analysis_window_batch_generator(self, corpus_words):
        windows_combination_vectors_in_sentence = []
        windows_combinations_in_sentence = []

        word_in_sentence_id = 0

        while word_in_sentence_id <= len(corpus_words) - self.__config.network_window_length + 1:
            # Is last window in sentence
            if (word_in_sentence_id + self.__config.network_window_length - 1 == len(corpus_words)) or \
                    (corpus_words[word_in_sentence_id + self.__config.network_window_length - 1] == self.analyses_processor.inverse_vocabulary[self.__config.marker_start_of_sentence]):

                padded_sentence_batch = self.__data_processor.pad_batch_list(
                    windows_combination_vectors_in_sentence,
                    self.__config.network_max_source_sequence_length,
                    self.__config.data_batch_size
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
            window_combinations_vector[-1] = self.analyses_processor.get_all_extra_info_vectors_for_word(last_word)

            combinations_in_window = list(itertools.product(*window_combinations_vector))

            vectorized_window_combinations = self.__data_processor.format_window_word_analyses(combinations_in_window)

            window_combinations = list(itertools.product(*window_analyses))

            windows_combination_vectors_in_sentence.extend(vectorized_window_combinations)
            windows_combinations_in_sentence.append(window_combinations)

            word_in_sentence_id += 1

    def feed_into_network(self, combinations_matrix):
        """combinations_matrix: numpy matrix with shape (self.__config.data_batch_size, self.__config.network_max_source_sequence_length)"""
        scores, all_output_sequences = self.__inference_session.run(
            [self.__inference_model.logits, self.__inference_model.output_sequences],
            feed_dict={
                self.__inference_model.placeholders.infer_inputs: combinations_matrix
            }
        )

        if self.__config.inference_decoder_type == 'beam_search':
            # for sequence in output_sequences:
            #     for row in sequence:
            #         print("\t".join(analyses_proce.analyses_processor.lookup_ids_to_features(row)))
            #     print('---')

            lifted_decoder_output_matrix = all_output_sequences + 1  # target scores
            masked_lifted_scores = np.multiply(lifted_decoder_output_matrix, scores)
            masked_scores = np.divide(masked_lifted_scores, lifted_decoder_output_matrix)
            beam_scores = np.nansum(masked_scores, axis=1)
            max_sequence_probabilities = np.max(beam_scores, axis=1)
            argmax_sequence_probabilities = np.argmax(beam_scores, axis=1)

            max_beams = []
            for sequence_id, argmax_column_id in enumerate(argmax_sequence_probabilities.tolist()):
                decoder_output_matrix = all_output_sequences[sequence_id]
                max_beams.append(decoder_output_matrix[:, argmax_column_id].tolist())

            max_beams = np.array(max_beams)

            all_output_sequences = max_beams
            scores = max_sequence_probabilities
        elif self.__config.inference_decoder_type == 'greedy':
            # inference_batch_probabilities = np.sum(np.log(np.max(logits, axis=2)), axis=1).tolist() # log for full sequence
            # inference_batch_probabilities = np.product(np.max(logits[0], axis=2), axis=1).tolist() # prob for full sequence
            max_log_probs_of_decoding_iterations = np.log(np.max(scores, axis=2))

            end_of_sentence_ids = np.argmax(all_output_sequences == self.__config.marker_end_of_sentence, axis=1)

            probabilities = []

            for target_output_id, end_of_sentence_id in enumerate(end_of_sentence_ids):
                probability = sum(max_log_probs_of_decoding_iterations[target_output_id][:end_of_sentence_id])  # log probability
                probabilities.append(probability)

            scores = probabilities

        return scores, all_output_sequences

    def __corpus_words_to_windows_and_probabilities(self, corpus_words):
        for windows_combinations_in_sentence, padded_sentence_batch in self.__create_analysis_window_batch_generator(corpus_words):
            # If the all the combinations was less or equal then data_batch_size
            if len(padded_sentence_batch) == self.__config.data_batch_size:
                padded_sentence_batch = np.matrix(padded_sentence_batch)
                scores, output_sequences = self.feed_into_network(padded_sentence_batch)

                horizontal_padding = np.zeros(shape=(self.__config.data_batch_size, self.__config.network_max_target_sequence_length - output_sequences.shape[1]))
                output_sequences = np.concatenate((output_sequences, horizontal_padding), axis=1)

                yield windows_combinations_in_sentence, scores, output_sequences

            else:
                inference_batch_pointer = 0
                probabilities = []
                output_sequences = []

                while inference_batch_pointer < len(padded_sentence_batch):
                    if inference_batch_pointer + self.__config.data_batch_size <= len(padded_sentence_batch):
                        padded_sentence_part_batch = np.matrix(padded_sentence_batch[inference_batch_pointer:inference_batch_pointer+self.__config.data_batch_size])
                    else:
                        padded_sentence_part_batch = np.matrix(
                            self.__data_processor.pad_batch_list(
                                padded_sentence_batch[inference_batch_pointer:],
                                self.__config.network_max_source_sequence_length,
                                self.__config.data_batch_size
                            )
                        )

                    scores, r_output_sequences = self.feed_into_network(padded_sentence_part_batch)

                    probabilities.extend(scores)

                    horizontal_padding = np.zeros(shape=(self.__config.data_batch_size, self.__config.network_max_target_sequence_length-r_output_sequences.shape[1]))
                    padded_output_sequences = np.concatenate((r_output_sequences, horizontal_padding), axis=1)
                    output_sequences.append(padded_output_sequences)

                    inference_batch_pointer+=self.__config.data_batch_size

                output_sequences = np.concatenate(tuple(partial_output_sequence for partial_output_sequence in output_sequences), axis=0)

                yield windows_combinations_in_sentence, probabilities, output_sequences

    @staticmethod
    def corpus_to_tokenized_sentences(corpus):
        """corpus: 'My name is Copus. And yours?'; Returns: [['My', 'name', 'is', 'Corpus', '.'], ['And', 'yours', '?']]"""
        corpus_temp_file = NamedTemporaryFile(delete=False)
        corpus_temp_file.write(corpus.encode())
        corpus_temp_file.close()

        hfst_pipe = subprocess.Popen(
            'quntoken ' + corpus_temp_file.name,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True
        )

        raw_output, err = hfst_pipe.communicate(corpus.encode())
        raw_decoded_output = raw_output.decode('utf-8')

        if len(err)>0:
            print(err.decode())

        root = ET.fromstring('<root>' + raw_decoded_output + '</root>')

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
        """tokenized_sentences: output of corpus_to_tokenized_sentences;
        Returns: [[(token, (disambiguated_analysis, log_probability, network_output)),...],...]"""
        flattened_corpus_words_with_SOS_before_sentences = []
        for tokenized_sentence in tokenized_sentences:
            flattened_corpus_words_with_SOS_before_sentences += [self.analyses_processor.inverse_vocabulary[self.__config.marker_start_of_sentence]] * (self.__config.network_window_length - 1) + tokenized_sentence

        sentence_id = 0
        disambiguated_analyses_for_sentences = []
        for disambiguated_analyses, _ in self.disambiguated_analyses_by_sentence_generator(flattened_corpus_words_with_SOS_before_sentences):
            disambiguated_analyses_for_sentences.append(zip(tokenized_sentences[sentence_id], disambiguated_analyses))
            sentence_id+=1

        return disambiguated_analyses_for_sentences

    def disambiguated_analyses_by_sentence_generator(self, corpus_words):
        """corpus_words: tokens in the corpus; Yields: disambiguated analyses and network output sequences for each sentence"""
        for windows_combinations_probabilities_output_sequences_in_sentence, all_output_sequences in self.__get_windows_combinations_probabilities_output_sequences_by_sentence_generator(corpus_words):
            viterbi_lists = []

            for window_combinations_probabilities_output_sequences in windows_combinations_probabilities_output_sequences_in_sentence:
                empty_window = True # If the sentence was too large to fit in the inference input matrix
                index_by_last_four_analyses = dict()
                for combination, probability, output_sequence in window_combinations_probabilities_output_sequences:

                    if self.__correct_combinations is not None and combination in self.__correct_combinations:
                        probability = 10000.0

                    empty_window = False

                    if combination[1:] in index_by_last_four_analyses.keys():
                        index_by_last_four_analyses[combination[1:]].append((combination, probability, output_sequence))
                    else:
                        index_by_last_four_analyses.setdefault(combination[1:], [(combination, probability, output_sequence)])

                if not empty_window:
                    reduced_groups_by_max_probability = []

                    for last_four_analyses, matching_combinations_with_probability in index_by_last_four_analyses.items():
                        combination_with_max_probability_in_group = max(matching_combinations_with_probability, key=itemgetter(1))
                        reduced_groups_by_max_probability.append(combination_with_max_probability_in_group)

                    viterbi_lists.append(reduced_groups_by_max_probability)

            disambiguated_combination_probability_output_sequence_tuples = []

            last_viterbi = viterbi_lists[-1]

            argmax_combination_probability_output_sequence_tuple = reduce(lambda max, combination_probability_tuple: combination_probability_tuple if combination_probability_tuple[1]>max[1] else max, last_viterbi, (['???'],-100))
            disambiguated_combination_probability_output_sequence_tuples.append(argmax_combination_probability_output_sequence_tuple)
            for viterbi in reversed(viterbi_lists[:-1]):
                argmax_combination_probability_output_sequence_tuple = next(filter(lambda combination_probability_tuple: combination_probability_tuple[0][1:]==argmax_combination_probability_output_sequence_tuple[0][:-1], viterbi))
                disambiguated_combination_probability_output_sequence_tuples.append(argmax_combination_probability_output_sequence_tuple)

            disambiguated_analyses = list(map(lambda cpo_tuple: (cpo_tuple[0][-1], cpo_tuple[1], cpo_tuple[2]), reversed(disambiguated_combination_probability_output_sequence_tuples)))
            yield disambiguated_analyses, all_output_sequences

    def __get_windows_combinations_probabilities_output_sequences_by_sentence_generator(self, corpus_words):
        for windows_combinations_in_sentence, probabilities_in_sentence, all_output_sequences in self.__corpus_words_to_windows_and_probabilities(corpus_words):
            windows_combinations_probabilities_output_sequences = []

            windows_combination_heights_in_sentence = list(map(lambda window_combinations: len(window_combinations), windows_combinations_in_sentence))

            for window_id, window_height in enumerate(windows_combination_heights_in_sentence):
                window_heights_so_far = sum(windows_combination_heights_in_sentence[:window_id])
                window_probabilities = []
                window_output_sequences = []
                for target_output_id in range(window_heights_so_far, window_heights_so_far+window_height):
                    try:
                        end_of_sentence_id = all_output_sequences[target_output_id].tolist().index(self.__config.marker_end_of_sentence)
                    except ValueError:
                        # No EOS in output sequence
                        end_of_sentence_id = len(all_output_sequences)
                    probability = probabilities_in_sentence[target_output_id]
                    window_probabilities.append(probability)
                    window_output_sequences.append("".join(self.analyses_processor.lookup_ids_to_features(all_output_sequences[target_output_id][:end_of_sentence_id])).replace("<PAD>", ""))

                window_combinations_in_sentence = [tuple(combination) for combination in windows_combinations_in_sentence[window_id]]

                windows_combinations_probabilities_output_sequences.append(zip(window_combinations_in_sentence, window_probabilities, window_output_sequences))

            # for window_combinations_with_probabilities in windows_combinations_with_probabilities:
            #     print('window')
            #     for combination, probability in window_combinations_with_probabilities:
            #         print(probability, combination)
            #
            # print('-----')
            yield windows_combinations_probabilities_output_sequences, all_output_sequences

    def evaluate_model(self, sentence_dicts, print_analyses = False):
        print('def evaluate_model(self, sentence_dicts, print_analyses = False):')
        disambiguation_accuracies = []
        print('\t#{sentences}: ', len(sentence_dicts))
        try:
            for sentence_id, sentence_dict in enumerate(sentence_dicts):
                if print_analyses:
                    print('\t\tSentence ', (sentence_id+1),'/', len(sentence_dicts))
                words_to_disambiguate = sentence_dict['word']  # Including <SOS>

                correct_analyses = sentence_dict['correct_analysis'][self.__config.network_window_length - 1:]

                # Faking correct combination probabilities to test Viterbi
                # self.__correct_combinations = []
                # for id in range(len(correct_analyses)-self.__config.network_window_length+1):
                #     correct_combination = []
                #     for window_analysis_id in range(id, id+self.__config.network_window_length):
                #         correct_combination.append(correct_analyses[window_analysis_id])
                #     self.__correct_combinations.append(tuple(correct_combination))

                disambiguated_sentence, all_output_sequences = next(self.disambiguated_analyses_by_sentence_generator(words_to_disambiguate))

                # Header print
                if print_analyses:
                    print('\t%-40s| %-40s| %-10s| %-s' % ("Correct analyses", "Disambiguated analyses", "P", "Network output"))
                    print('\t'+ ('-' * 120))

                # Disambiguation accuracy
                matching_analyses = 0
                for i, apo_tuple in enumerate(disambiguated_sentence):
                    if print_analyses:
                        print('\t%-40s| %-40s| %-10.4g| %-s' % (correct_analyses[i], apo_tuple[0], apo_tuple[1], apo_tuple[2]))
                    if apo_tuple[0] == correct_analyses[i]:
                        matching_analyses += 1

                disambiguation_accuracy = 100 * matching_analyses / len(disambiguated_sentence)
                disambiguation_accuracies.append(disambiguation_accuracy)

                if disambiguation_accuracy>97.5:
                    for i, apo_tuple in enumerate(disambiguated_sentence):
                        print('\t%-40s| %-40s| %-10.4g| %-s' % (correct_analyses[i], apo_tuple[0], apo_tuple[1], apo_tuple[2]))


                # Sentence result
                if print_analyses:
                    print('\t' + ('-' * 120))
                    print('\t>> Disambiguation accuracy: %-6.2f\n' % (disambiguation_accuracy))
                else:
                    print('\t\tSentence: %8d/%-8d Disambiguation accuracy: %-6.2f' % (sentence_id + 1, len(sentence_dicts), disambiguation_accuracy))

                if (sentence_id+1) % 5 == 0:
                    average_disambiguation_accuracy = sum(disambiguation_accuracies) / len(disambiguation_accuracies)
                    print('\tDisambiguation accuracies: min - %-6.2f max - %-6.2f avg - %6.2f' % (min(disambiguation_accuracies), max(disambiguation_accuracies), average_disambiguation_accuracy))

        except KeyboardInterrupt:
            # fig = plt.figure()
            # ax1 = fig.add_subplot(111)
            #
            # ax1.scatter(range(len(disambiguation_accuracies)), disambiguation_accuracies, s=10, c='b', marker="s", label='disambiguation accuracies')
            # ax1.scatter(range(len(network_output_accuracies)), network_output_accuracies, s=10, c='b', marker="o", label='network output accuracies')
            # plt.legend(loc='upper left')
            #
            # plt.savefig('inference_accuracy.png')
            print('Interrupted by user.')

        if len(disambiguation_accuracies)>0:
            print('Writing output CSV...')
            with open('disambiguation_accuracies_output_'+self.__config.data_example_resolution+'.csv', 'w') as f:
                f.writelines(list(map(lambda l: str(l) + os.linesep, disambiguation_accuracies)))
            print('>>>> Disambiguation accuracies\t-\tmin: %-12.2f max: %-12.2f avg: %-12.2f' % (
                min(disambiguation_accuracies), max(disambiguation_accuracies), sum(disambiguation_accuracies) / len(disambiguation_accuracies)))
        print('=' * 100)