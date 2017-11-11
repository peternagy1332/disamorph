import argparse
import os
from random import shuffle, seed

from config import ModelConfiguration
from data_processing.analyses_processor import AnalysesProcessor
from data_processing.data_processor import DataProcessor
from disambiguator import Disambiguator
from utils import Utils


def main():
    parser = argparse.ArgumentParser(description='Hungarian morphological disambiguator')
    parser.add_argument('-cfg', '--default-config', default=os.path.join('configs', 'default_config.yaml'))
    parser.add_argument('-m', '--model-directory', required=True)

    model_configuration = ModelConfiguration(parser)
    utils = Utils(model_configuration)
    utils.start_stopwatch()
    utils.redirect_stdout('main-evaluation')

    model_configuration.printConfig()

    # Setting seed
    if model_configuration.data_random_seed is not None:
        seed(model_configuration.data_random_seed)

    # Loading train data
    analyses_processor = AnalysesProcessor(model_configuration)
    data_processor = DataProcessor(model_configuration, analyses_processor)

    sentence_dataframes = data_processor.get_sentence_dicts(False)

    test_dataframe_id = int(round(len(sentence_dataframes)*model_configuration.test_sentences_rate))
    test_dataframes = sentence_dataframes[-test_dataframe_id:]

    train_dataframes = sentence_dataframes[:-test_dataframe_id]
    if model_configuration.train_shuffle_sentences:
        shuffle(train_dataframes)

    # Evaluating model
    disambiguator = Disambiguator(model_configuration, analyses_processor)

    print('Evaluating model on train dataset...')
    disambiguator.evaluate_model(train_dataframes[:1000])

    print('Evaluating model on test dataset...')
    disambiguator.evaluate_model(test_dataframes)

    utils.stop_stopwatch_and_print_running_time()

if __name__ == '__main__':
    main()
