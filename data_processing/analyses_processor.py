import re
import subprocess
import os


class AnalysesProcessor(object):
    def __init__(self, model_configuration):
        self.__config = model_configuration

        if self.__config.train_rebuild_vocabulary_file:
            self.vocabulary_file = open(os.path.join('data', 'vocabulary_cache_long.tsv'), 'w', encoding='utf8')

        self.__read_and_build_vocabulary()
        self.__hfst_cache_vectorized = dict()
        self.__hfst_cache = dict()

    def __read_and_build_vocabulary(self):
        print('def __read_and_build_vocabulary(self):')
        features = []
        with open(self.__config.train_vocabulary_cache, encoding='utf-8') as f: features.extend(f.read().splitlines())
        self.vocabulary = dict(zip(features, range(self.__config.vocabulary_start_index, len(features) + self.__config.vocabulary_start_index+1)))

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

            if '+?' in raw_decoded_output:
                vectorized_analyses_for_word = [[self.vocabulary.get(raw_decoded_output.rstrip(), self.__config.marker_unknown)]]
            else:
                raw_tapes_of_analyses = raw_decoded_output.splitlines()
                raw_tapes_of_analyses = [raw_tape_of_analysis.rstrip() for raw_tape_of_analysis in raw_tapes_of_analyses]
                tapes_of_analyses = [self.raw_tape_to_tape(raw_tape) for raw_tape in raw_tapes_of_analyses]
                morphemes_and_tags_of_analyes = [self.tape_to_morphemes_and_tags(tape) for tape in tapes_of_analyses]

                vectorized_analyses_for_word = [self.lookup_morphemes_and_tags_to_ids(morphemes_and_tags) for morphemes_and_tags in morphemes_and_tags_of_analyes]

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

            if '+?' in raw_decoded_output:
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
        return tape


    def tape_to_morphemes_and_tags(self, tape):
        morphemes_and_tags = []
        morpheme_buffer = ''
        for cell in tape:
            if re.match(r'\[[^]]*\]', cell, re.UNICODE):
                if len(morpheme_buffer)>0:
                    morphemes_and_tags.append(morpheme_buffer)
                    morpheme_buffer = ''
                morphemes_and_tags.append(cell)
            else:
                morpheme_buffer+=cell

        if len(morpheme_buffer) >0:
            morphemes_and_tags.append(morpheme_buffer)

        if self.__config.train_rebuild_vocabulary_file:
            for morpheme_or_tag in morphemes_and_tags:
                self.vocabulary_file.write(morpheme_or_tag+os.linesep)

        return morphemes_and_tags

    def lookup_morphemes_and_tags_to_ids(self, morphemes_and_tags_list):
        # TODO: unk tömbben
        return [self.vocabulary.get(morpheme_or_tag, self.__config.marker_unknown) for morpheme_or_tag in morphemes_and_tags_list]

    def lookup_ids_to_morphemes_and_tags(self, ids_list):
        return [self.inverse_vocabulary.get(id, self.inverse_vocabulary[self.__config.marker_unknown]) for id in ids_list]
