import glob
import os
from collections import namedtuple
import pandas as pd
import numpy as np
import re


class MorphDisamTrainer(object):
    def __init__(self, train_files_paths, window_size=5):
        self.__vocabulary = self.__read_features_to_vocabulary(train_files_paths.tags, train_files_paths.morphemes)
        self.__corpus_dataframe = self.__read_corpus_dataframe(train_files_paths.corpus)
        self.__window_size = window_size

    def __read_features_to_vocabulary(self, file_tags, file_morphemes):
        features = []
        with open(file_tags, encoding='utf-8') as f: features.extend(f.read().splitlines())
        with open(file_morphemes, encoding='utf-8') as f: features.extend(f.read().splitlines())
        return dict(zip(features, range(1, len(features) + 1)))

    def __read_corpus_dataframe(self, path_corpuses):
        return pd.concat(
            (pd.read_csv(f, sep='\t', usecols=[4], skip_blank_lines=True, header=None, nrows=2, names=['analyses'])
             for f in glob.glob(path_corpuses)))

    def __analysis_to_vector(self, analysis):
        root = re.search(r'\w+', analysis, re.UNICODE).group(0)
        tags = re.findall(r'\[[^]]*\]', analysis, re.UNICODE)
        return np.matrix([self.__vocabulary[root]] + [self.__vocabulary[tag] for tag in tags])

    def generate_batch(self, batch_size):
        for analysis in self.__corpus_dataframe['analyses']:
            print(self.__analysis_to_vector(analysis))


def main():
    TrainFilesPaths = namedtuple('TrainFilesPaths', ['tags', 'morphemes', 'corpus'])

    morph_disam_trainer = MorphDisamTrainer(TrainFilesPaths(
        tags=os.path.join('data', 'tags.txt'),
        morphemes=os.path.join('data', 'morphemes.txt'),
        corpus=os.path.join('data', 'szeged', '*')
    ))

    init_batch = morph_disam_trainer.generate_batch(batch_size=10)

    print(init_batch)


if __name__ == '__main__':
    main()
