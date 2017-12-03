import argparse

from __init__ import ModelConfiguration
from disambiguator import Disambiguator


def main():
    parser = argparse.ArgumentParser(description='Hungarian morphological disambiguator')
    parser.add_argument('-cfg', '--default-config', default=None)
    parser.add_argument('-m', '--model-directory', required=True)
    parser.add_argument('-t', '--use-train-model', default=False, action='store_true')

    model_configuration = ModelConfiguration(parser)

    disambiguator = Disambiguator(model_configuration)

    tokenized_sentences = disambiguator.corpus_to_tokenized_sentences(input())

    for correct_analyses in disambiguator.disambiguate_tokenized_sentences(tokenized_sentences):
        for (token, (disambiguated_analysis, log_probability, network_output)) in correct_analyses:
            print('%s\t%s' % (token, disambiguated_analysis))
        print()

if __name__ == '__main__':
    main()