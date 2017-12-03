import operator
import os
import re
import subprocess
import sys

from disamorph.utils import Colors


class AnalysesProcessor(object):
    def __init__(self, model_configuration):
        self.__config = model_configuration

        self.vocabulary = dict()
        self.inverse_vocabulary = dict()

        self.__read_or_build_vocabulary()
        self.__hfst_cache_vectorized = dict()
        self.__hfst_cache_feature_list_analyses = dict()
        self.__hfst_cache_analyses = dict()

    def __read_or_build_vocabulary(self):
        features = []
        if os.path.isfile(self.__config.data_vocabulary_file):
            with open(self.__config.data_vocabulary_file, 'r', encoding='utf-8') as f: features.extend(f.read().splitlines())
            self.vocabulary = dict(zip(features, range(len(features))))
        else:
            print('\tVocabulary file not found ('+self.__config.data_vocabulary_file+'), rebuild is going to be performed.')
            self.__config.train_rebuild_vocabulary_file = True

        self.inverse_vocabulary = {v: k for k, v in self.vocabulary.items()}

        self.inverse_vocabulary[self.__config.marker_padding] = '<PAD>'
        self.inverse_vocabulary[self.__config.marker_analysis_divider] = '<DIV>'
        self.inverse_vocabulary[self.__config.marker_start_of_sentence] = '<SOS>'
        self.inverse_vocabulary[self.__config.marker_end_of_sentence] = '<EOS>'
        self.inverse_vocabulary[self.__config.marker_unknown] = '<UNK>'
        self.inverse_vocabulary[self.__config.marker_go] = '<GO>'

    def get_feature_list_analyses_for_word(self, word):
        """Output: har:[[a b c TAG],...] or morph:[[abc TAG,...]"""

        artificial_tags = [
            self.inverse_vocabulary[self.__config.marker_padding],
            self.inverse_vocabulary[self.__config.marker_analysis_divider],
            self.inverse_vocabulary[self.__config.marker_start_of_sentence],
            self.inverse_vocabulary[self.__config.marker_end_of_sentence],
            self.inverse_vocabulary[self.__config.marker_unknown]
        ]

        if word in artificial_tags:
            return [[word]]

        if word in self.__hfst_cache_feature_list_analyses.keys():
            return self.__hfst_cache_feature_list_analyses[word]

        hfst_pipe = subprocess.Popen(
            'hfst-lookup --pipe-mode=input --cascade=composition --xfst=print-pairs --xfst=print-space -s ' + self.__config.inference_transducer_path + ' | cut -f2 | grep .',
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True
        )

        raw_output, err = hfst_pipe.communicate(word.encode())
        raw_decoded_output = raw_output.decode('utf-8')

        if len(err) > 0:
            print(err.decode())

        raw_tapes_of_analyses = raw_decoded_output.splitlines()

        feature_list_analyses = []
        for raw_tape_of_analysis in raw_tapes_of_analyses:
            tape = self.raw_tape_to_tape(raw_tape_of_analysis)
            if self.__config.data_example_resolution == 'character':
                feature_list_analyses.append(tape)
            elif self.__config.data_example_resolution == 'morpheme':
                feature_list_analyses.append(self.tape_to_morphemes_and_tags(tape))

        self.__hfst_cache_feature_list_analyses.setdefault(word, feature_list_analyses)

        return feature_list_analyses


    def get_lookedup_feature_list_analyses_for_word(self, word):
        """Output: char:[[1 2 3 4],...] or morph:[[1 2,...]"""
        artificial_tags = [
            self.inverse_vocabulary[self.__config.marker_padding],
            self.inverse_vocabulary[self.__config.marker_analysis_divider],
            self.inverse_vocabulary[self.__config.marker_start_of_sentence],
            self.inverse_vocabulary[self.__config.marker_end_of_sentence],
            self.inverse_vocabulary[self.__config.marker_unknown]
        ]

        if word in artificial_tags:
            return [[self.vocabulary[word]]]

        if word in self.__hfst_cache_vectorized.keys():
            vectorized_analyses = self.__hfst_cache_vectorized[word]
        else:

            feature_list_analyses_for_word = self.get_feature_list_analyses_for_word(word)

            vectorized_analyses = []

            for feature_list_analysis in feature_list_analyses_for_word:
                vectorized_analyses.append(self.lookup_features_to_ids(feature_list_analysis))

            self.__hfst_cache_vectorized.setdefault(word, vectorized_analyses)

        return vectorized_analyses

    def get_analyses_list_for_word(self, word):
        """Output: char:[[abcTAG],...] or morph:[[abcTAG,...]"""
        artificial_tags = [
            self.inverse_vocabulary[self.__config.marker_padding],
            self.inverse_vocabulary[self.__config.marker_analysis_divider],
            self.inverse_vocabulary[self.__config.marker_start_of_sentence],
            self.inverse_vocabulary[self.__config.marker_end_of_sentence],
            self.inverse_vocabulary[self.__config.marker_unknown]
        ]

        if word in artificial_tags:
            return [word]

        if word in self.__hfst_cache_analyses.keys():
            analyses = self.__hfst_cache_analyses[word]
        else:

            feature_list_analyses_for_word = self.get_feature_list_analyses_for_word(word)

            analyses = []

            for feature_list_analysis in feature_list_analyses_for_word:
                analyses.append("".join(feature_list_analysis))

            self.__hfst_cache_analyses.setdefault(word, analyses)

        return analyses

    def raw_tape_to_tape(self, raw_tape):
        tape = []

        if raw_tape[-2:] == '+?':
            if self.__config.data_example_resolution == 'character':
                for character in raw_tape[:-2]:
                    tape.append(character)
            elif self.__config.data_example_resolution == 'morpheme':
                tape.append(raw_tape[:-2])
        else:
            tape_parts = raw_tape.split(' ')
            for part in tape_parts:
                tape_outputs = part.split(':')
                if len(tape_outputs) > 1:
                        tape.append(":".join(tape_outputs[1:]))

            if self.__config.train_rebuild_vocabulary_file and self.__config.data_example_resolution == 'character':
                for character_or_tag in tape:
                    self.inverse_vocabulary.setdefault(len(self.inverse_vocabulary), character_or_tag)

        return tape

    def tape_to_morphemes_and_tags(self, tape):
        morphemes_and_tags = []
        morpheme_buffer = ''
        for cell in tape:
            if re.match(r'\[[^]]*\]', cell, re.UNICODE):
                if len(morpheme_buffer) > 0:
                    morphemes_and_tags.append(morpheme_buffer)
                    morpheme_buffer = ''
                morphemes_and_tags.append(cell)
            else:
                morpheme_buffer+=cell

        if len(morpheme_buffer) > 0:
            morphemes_and_tags.append(morpheme_buffer)

        if self.__config.train_rebuild_vocabulary_file:
            for morpheme_or_tag in morphemes_and_tags:
                self.inverse_vocabulary.setdefault(len(self.inverse_vocabulary), morpheme_or_tag)

        return morphemes_and_tags

    def extend_vocabulary(self, feature):
        if self.__config.train_rebuild_vocabulary_file:
            self.inverse_vocabulary.setdefault(len(self.inverse_vocabulary), feature)

    def write_vocabulary_file(self):
        print('def write_vocabulary_file(self):')
        if self.__config.train_rebuild_vocabulary_file:
            # So artificial tags will be first
            ordered_feature_list = list(map(lambda x: x[1], sorted(self.inverse_vocabulary.items(), key=operator.itemgetter(0))))
            unique_ordered_feature_list = sorted(set(ordered_feature_list), key=ordered_feature_list.index)

            with open(self.__config.data_vocabulary_file, 'w+', encoding='utf8') as vocabulary_file:
                vocabulary_file.writelines(map(lambda feature: feature+os.linesep, unique_ordered_feature_list))
            print(Colors.YELLOW + '\tVocabulary file flushed:',self.__config.data_vocabulary_file)

            print(Colors.LIGHTRED + '\tPlease restart the application.')
            sys.exit()
        else:
            print(Colors.YELLOW + '\tUsing existing vocabulary file:',self.__config.data_vocabulary_file)

    def get_extra_info_vector(self, feature_list_analysis):
        """Input: char:[a b c TAG1 d e TAG2 ...] or morph:[abc TAG1 de TAG2 ...]
          Output: char:[a b c] or morph: [abc] (IDs until first tag or end of list)"""
        if self.__config.data_example_resolution == 'character':
            morphemes_and_tags = self.tape_to_morphemes_and_tags(feature_list_analysis)
            first_morpheme = morphemes_and_tags[0]
            extra_info = []
            for char in first_morpheme:
                extra_info.append(self.vocabulary.get(char, self.__config.marker_unknown))
        else:
            morphemes_and_tags = feature_list_analysis
            first_morpheme = morphemes_and_tags[0]
            extra_info = [self.vocabulary.get(first_morpheme, self.__config.marker_unknown)]

        return extra_info


    def get_all_extra_info_vectors_for_word(self, word):

        feature_list_analyses = self.get_feature_list_analyses_for_word(word)

        extra_infos = []

        for feature_list_analysis in feature_list_analyses:
            extra_info = self.get_extra_info_vector(feature_list_analysis)
            extra_infos.append(extra_info)

        return extra_infos

    def lookup_features_to_ids(self, feature_list):
        return [self.vocabulary.get(feature, self.__config.marker_unknown) for feature in feature_list]

    def lookup_ids_to_features(self, ids_list):
        return [self.inverse_vocabulary.get(id, self.inverse_vocabulary[self.__config.marker_unknown]) for id in ids_list]
