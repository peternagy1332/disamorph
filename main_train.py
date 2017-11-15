import argparse
from config import ModelConfiguration
from data_processing.analyses_processor import AnalysesProcessor
from data_processing.data_processor import DataProcessor
from model.model_train import BuildTrainModel
from seq2seq_trainer import Seq2SeqTrainer
from utils import Utils


def main():
    parser = argparse.ArgumentParser(description='Hungarian morphological disambiguator')
    parser.add_argument('-dcfg', '--default-config', default=None)
    parser.add_argument('-m', '--model-directory', default=None)

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

    # Building graph
    build_train_model = BuildTrainModel(model_configuration, analyses_processor.vocabulary, analyses_processor.inverse_vocabulary)

    model = build_train_model.create_model()

    # Begin training
    morph_disam_trainer = Seq2SeqTrainer(build_train_model.train_graph, model_configuration, model, utils)


    morph_disam_trainer.train(data_processor, sentence_dicts)
    utils.print_elapsed_time()


if __name__ == '__main__':
    main()
