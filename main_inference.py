import argparse
from disamorph import ModelConfiguration
from disamorph.disambiguator import Disambiguator


def main():
    parser = argparse.ArgumentParser(description='Disamorph: A Hungarian morphological disambiguator using sequence-to-sequence neural networks')
    parser.add_argument('-m', '--model-directory', required=True, help='Path to the model directory.')
    parser.add_argument('-t', '--use-train-model', default=False, action='store_true', help='Whether to use the train insted of the validation model.')

    model_configuration = ModelConfiguration(parser)

    disambiguator = Disambiguator(model_configuration)

    tokenized_sentences = disambiguator.corpus_to_tokenized_sentences(input())

    for correct_analyses in disambiguator.disambiguate_tokenized_sentences(tokenized_sentences):
        for (token, (disambiguated_analysis, log_probability, network_output)) in correct_analyses:
            print('%s\t%s' % (token, disambiguated_analysis))
        print()

if __name__ == '__main__':
    main()