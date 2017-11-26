import argparse
import os

from config import ModelConfiguration
from data_processing.analyses_processor import AnalysesProcessor
from data_processing.data_processor import DataProcessor
from disambiguator import Disambiguator
from utils import Utils


def main():
    parser = argparse.ArgumentParser(description='Hungarian morphological disambiguator')
    parser.add_argument('-dcfg', '--default-config', default=os.path.join('default_configs', 'character.yaml'))
    parser.add_argument('-m', '--model-directory', required=True)
    parser.add_argument('-t', '--use-train-model', default=False, action='store_true')

    model_configuration = ModelConfiguration(parser)
    model_configuration.train_shuffle_examples_in_batches = False
    model_configuration.train_shuffle_sentences = False

    utils = Utils(model_configuration)
    utils.start_stopwatch()
    utils.redirect_stdout('main-evaluation')

    model_configuration.printConfig()

    # Data utilities
    analyses_processor = AnalysesProcessor(model_configuration)
    data_processor = DataProcessor(model_configuration, analyses_processor)

    train_sentence_dicts, validation_sentence_dicts, test_sentence_dicts = data_processor.get_split_sentence_dicts(False)
    utils.print_elapsed_time()

    # Evaluating model
    disambiguator = Disambiguator(model_configuration, analyses_processor)

    print('Evaluating model on train dataset...')
    disambiguator.evaluate_model(train_sentence_dicts[:1000], False)
    utils.print_elapsed_time()

    print('Evaluating model on validation dataset...')
    disambiguator.evaluate_model(validation_sentence_dicts, False)
    utils.print_elapsed_time()

    print('Evaluating model on test dataset...')
    disambiguator.evaluate_model(test_sentence_dicts, False)
    utils.print_elapsed_time()

if __name__ == '__main__':
    main()
