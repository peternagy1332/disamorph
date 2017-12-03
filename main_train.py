import argparse
from disamorph import ModelConfiguration, Utils
from disamorph.data_processing.analyses_processor import AnalysesProcessor
from disamorph.data_processing.data_processor import DataProcessor
from disamorph.seq2seq_trainer import Seq2SeqTrainer


def main():
    parser = argparse.ArgumentParser(description='Disamorph: A Hungarian morphological disambiguator using sequence-to-sequence neural networks.')
    parser.add_argument('-dcfg', '--default-config', default=None, help='If provided, a new model will be trained with this config. Has priority over --model-directory.')
    parser.add_argument('-m', '--model-directory', default=None, help='If provided, the training of an existing model will be continued. If --default-config is also present, the new model will be saved to this path.')
    parser.add_argument('-t', '--use-train-model', default=False, action='store_true', help='On model continuation, for defining whether to continue the train insted of the validation model.')
    args = parser.parse_args()

    model_configuration = ModelConfiguration(args)
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
