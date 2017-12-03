import argparse

from __init__ import ModelConfiguration
from data_processing.analyses_processor import AnalysesProcessor
from data_processing.data_processor import DataProcessor
from seq2seq_trainer import Seq2SeqTrainer
from utils import Utils


def main():
    parser = argparse.ArgumentParser(description='Hungarian morphological disambiguator')
    parser.add_argument('-dcfg', '--default-config', default=None)
    parser.add_argument('-m', '--model-directory', default=None)
    parser.add_argument('-t', '--use-train-model', default=False, action='store_true')

    model_configuration = ModelConfiguration(parser)
    utils = Utils(model_configuration)
    utils.start_stopwatch()
    utils.redirect_stdout('main-train')

    model_configuration.printConfig()

    # Data processing utilities
    analyses_processor = AnalysesProcessor(model_configuration)
    utils.print_elapsed_time()

    data_processor = DataProcessor(model_configuration, analyses_processor)

    # Reading corpus
    sentence_dicts = data_processor.get_sentence_dicts()

    utils.print_elapsed_time()

    # Begin training
    morph_disam_trainer = Seq2SeqTrainer(analyses_processor, model_configuration, utils)


    morph_disam_trainer.train(data_processor, sentence_dicts)
    utils.print_elapsed_time()


if __name__ == '__main__':
    main()
