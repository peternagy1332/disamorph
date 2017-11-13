import re
import subprocess
import os

import sys


class AnalysesProcessor(object):
    def __init__(self, model_configuration):
        self.__config = model_configuration

        self.__vocabulary_set = set()

        self.__read_and_build_vocabulary()
        self.__hfst_cache_vectorized = dict()
        self.__hfst_cache = dict()

    def __read_and_build_vocabulary(self):
        print('def __read_and_build_vocabulary(self):')
        features = []
        if os.path.isfile(self.__config.data_vocabulary_file):
            with open(self.__config.data_vocabulary_file, 'r', encoding='utf-8') as f: features.extend(f.read().splitlines())
            self.vocabulary = dict(zip(features, range(self.__config.vocabulary_start_index, len(features) + self.__config.vocabulary_start_index+1)))
        else:
            print('\tVocabulary file not found ('+self.__config.data_vocabulary_file+'), rebuild is going to be performed.')
            self.__config.train_rebuild_vocabulary_file = True
            self.vocabulary = dict()

        self.vocabulary['<SOS>'] = self.__config.marker_start_of_sentence

        self.inverse_vocabulary = {v: k for k, v in self.vocabulary.items()}

        self.inverse_vocabulary[self.__config.marker_unknown] = '<UNK>'
        self.inverse_vocabulary[self.__config.marker_padding] = '<PAD>'
        self.inverse_vocabulary[self.__config.marker_end_of_sentence] = '<EOS>'
        self.inverse_vocabulary[self.__config.marker_start_of_sentence] = '<SOS>'
        self.inverse_vocabulary[self.__config.marker_analysis_divider] = '<DIV>'
        self.inverse_vocabulary[self.__config.marker_go] = '<GO>'

    def get_analyses_vector_list_for_word(self, word):
        artificial_tags = [
            self.inverse_vocabulary[self.__config.marker_start_of_sentence],
            self.inverse_vocabulary[self.__config.marker_end_of_sentence],
            self.inverse_vocabulary[self.__config.marker_padding],
            self.inverse_vocabulary[self.__config.marker_unknown],
            self.inverse_vocabulary[self.__config.marker_analysis_divider]
        ]

        if word in artificial_tags:
            return [[self.vocabulary[word]]]

        if word in self.__hfst_cache_vectorized.keys():
            vectorized_analyses_for_word = self.__hfst_cache_vectorized[word]
        else:
            hfst_pipe = subprocess.Popen(
                'hfst-lookup --pipe-mode=input --cascade=composition --xfst=print-pairs --xfst=print-space -s ' + self.__config.transducer_path + ' | cut -f2 | grep .',
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True
            )

            raw_output, err = hfst_pipe.communicate(word.encode())
            raw_decoded_output = raw_output.decode('utf-8')

            if len(err)>0:
                print(err.decode())

            vectorized_analyses_for_word = None

            if '+?' == raw_decoded_output[-2:]:
                unknown_analysis = raw_decoded_output.rstrip()
                if self.__config.data_example_resolution == 'morpheme':
                    vectorized_analyses_for_word = [[self.vocabulary.get(unknown_analysis, self.__config.marker_unknown)]]
                elif self.__config.data_example_resolution == 'character':
                    vectorized_analyses_for_word = []
                    vectorized_analysis_for_word = []
                    for character in unknown_analysis:
                        vectorized_analysis_for_word.append(self.vocabulary.get(character, self.__config.marker_unknown))
                    vectorized_analyses_for_word.append(vectorized_analysis_for_word)
            else:
                raw_tapes_of_analyses = raw_decoded_output.splitlines()
                raw_tapes_of_analyses = [raw_tape_of_analysis.rstrip() for raw_tape_of_analysis in raw_tapes_of_analyses]
                tapes_of_analyses = [self.raw_tape_to_tape(raw_tape) for raw_tape in raw_tapes_of_analyses]

                if self.__config.data_example_resolution == 'morpheme':
                    morphemes_and_tags_of_analyes = [self.tape_to_morphemes_and_tags(tape) for tape in tapes_of_analyses]
                    vectorized_analyses_for_word = [self.lookup_morphemes_and_tags_to_ids(morphemes_and_tags) for morphemes_and_tags in morphemes_and_tags_of_analyes]
                elif self.__config.data_example_resolution == 'character':
                    vectorized_analyses_for_word = []
                    for tape in tapes_of_analyses:
                        vectorized_analysis_for_word = self.lookup_morphemes_and_tags_to_ids(tape)
                        vectorized_analyses_for_word.append(vectorized_analysis_for_word)

            if vectorized_analyses_for_word is None:
                raise ValueError('Cannot vectorize HFST output "'+raw_decoded_output+'". Check data_example_resolution in config!')

            self.__hfst_cache_vectorized.setdefault(word, vectorized_analyses_for_word)

        return vectorized_analyses_for_word

    def get_analyses_list_for_word(self, word):
        artificial_tags = [
            self.inverse_vocabulary[self.__config.marker_start_of_sentence],
            self.inverse_vocabulary[self.__config.marker_end_of_sentence],
            self.inverse_vocabulary[self.__config.marker_padding],
            self.inverse_vocabulary[self.__config.marker_unknown],
            self.inverse_vocabulary[self.__config.marker_analysis_divider]
        ]

        if word in artificial_tags:
            return [word]

        if word in self.__hfst_cache.keys():
            analyses_for_word = self.__hfst_cache[word]
        else:
            hfst_pipe = subprocess.Popen(
                'hfst-lookup --pipe-mode=input --cascade=composition --xfst=print-pairs --xfst=print-space -s ' + self.__config.transducer_path + ' | cut -f2 | grep .',
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True
            )

            raw_output, err = hfst_pipe.communicate(word.encode())
            raw_decoded_output = raw_output.decode('utf-8')

            if len(err)>0:
                print(err.decode())

            if '+?' == raw_decoded_output[-2:]:
                analyses_for_word = [raw_decoded_output.rstrip()]
            else:
                raw_tapes_of_analyses = raw_decoded_output.splitlines()
                raw_tapes_of_analyses = [raw_tape_of_analysis.rstrip() for raw_tape_of_analysis in raw_tapes_of_analyses]

                tapes_of_analyses = [self.raw_tape_to_tape(raw_tape) for raw_tape in raw_tapes_of_analyses]

                morphemes_and_tags_of_analyes = [self.tape_to_morphemes_and_tags(tape) for tape in tapes_of_analyses]

                analyses_for_word = ["".join(morphemes_and_tags) for morphemes_and_tags in morphemes_and_tags_of_analyes]

            self.__hfst_cache.setdefault(word, analyses_for_word)

        return analyses_for_word

    def raw_tape_to_tape(self, raw_tape):
        tape = []
        tape_parts = raw_tape.split(' ')
        for part in tape_parts:
            tape_outputs = part.split(':')
            if len(tape_outputs) > 1:
                    tape.append(":".join(tape_outputs[1:]))

        if self.__config.train_rebuild_vocabulary_file and self.__config.data_example_resolution == 'character':
            for character_or_tag in tape:
                self.__vocabulary_set.add(character_or_tag)

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
                self.__vocabulary_set.add(morpheme_or_tag)

        return morphemes_and_tags

    def extend_vocabulary(self, feature):
        if self.__config.train_rebuild_vocabulary_file:
            self.__vocabulary_set.add(feature)

    def write_vocabulary_file(self):
        print('def write_vocabulary_file(self):')
        if self.__config.train_rebuild_vocabulary_file:
            with open(self.__config.data_vocabulary_file, 'w', encoding='utf8') as vocabulary_file:
                vocabulary_file.writelines(map(lambda feature: feature+os.linesep, sorted(self.__vocabulary_set)))
            print('\tVocabulary file flushed:',self.__config.data_vocabulary_file)
            print('\tPlease restart the application.')
            sys.exit()
        else:
            print('\tUsing existing vocabulary file:',self.__config.data_vocabulary_file)


    def lookup_morphemes_and_tags_to_ids(self, morphemes_and_tags_list):
        return [self.vocabulary.get(morpheme_or_tag, self.__config.marker_unknown) for morpheme_or_tag in morphemes_and_tags_list]

    def lookup_ids_to_morphemes_and_tags(self, ids_list):
        return [self.inverse_vocabulary.get(id, self.inverse_vocabulary[self.__config.marker_unknown]) for id in ids_list]
