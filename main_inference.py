import argparse
import os

from config import ModelConfiguration
from disambiguator import Disambiguator
from utils import Utils


def main():
    parser = argparse.ArgumentParser(description='Hungarian morphological disambiguator')
    parser.add_argument('-cfg', '--default-config', default=os.path.join('default_configs', 'character.yaml'))
    parser.add_argument('-m', '--model-directory', required=True)

    model_configuration = ModelConfiguration(parser)
    utils = Utils(model_configuration)
    utils.start_stopwatch()
    #utils.redirect_stdout('main-inference')

    disambiguator = Disambiguator(model_configuration)

    print("Enter corpus: ")
    corpus = input()
    tokenized_sentences = disambiguator.corpus_to_tokenized_sentences(corpus)

    print('%-30s%30s' % ('Word in corpus', 'Disambiguated analysis'))
    print('=' * 60)
    sentence_id = 0
    for correct_analyses in disambiguator.disambiguate_tokenized_sentences(tokenized_sentences):
        for word, analysis in zip(tokenized_sentences[sentence_id], correct_analyses):
            print('%-30s%-30s' % (word, analysis))
        print('-' * 60)
        sentence_id+=1

    utils.print_elapsed_time()

if __name__ == '__main__':
    main()