import argparse
from random import seed
from config import ModelConfiguration
from data_processing.analyses_processor import AnalysesProcessor
from data_processing.data_processor import DataProcessor
from model.model_train import BuildTrainModel
from seq2seq_trainer import Seq2SeqTrainer
from utils import Utils


def main():
    parser = argparse.ArgumentParser(description='Hungarian morphological disambiguator')
    parser.add_argument('-cfg', '--default-config', default=None)
    parser.add_argument('-m', '--model-directory', default=None)

    model_configuration = ModelConfiguration(parser)
    utils = Utils(model_configuration)
    utils.start_stopwatch()
    utils.redirect_stdout('main-train')

    model_configuration.printConfig()

    # Setting seed
    if model_configuration.data_random_seed is not None:
        seed(model_configuration.data_random_seed)

    # Loading sentence dataframes
    analyses_processor = AnalysesProcessor(model_configuration)
    data_processor = DataProcessor(model_configuration, analyses_processor)

    sentence_daraframes = data_processor.get_sentence_dicts()

    # All dataframes -> train dataframes
    test_dataframe_id = int(round(len(sentence_daraframes)*model_configuration.test_sentences_rate))
    # test_dataframes = sentence_daraframes[-test_dataframe_id:]
    train_dataframes = sentence_daraframes[:-test_dataframe_id]


    # Building graph
    build_train_model = BuildTrainModel(model_configuration,
                                        analyses_processor.vocabulary,
                                        analyses_processor.inverse_vocabulary)

    model = build_train_model.create_model()

    # Begin training
    morph_disam_trainer = Seq2SeqTrainer(build_train_model.train_graph, model_configuration, model)

    morph_disam_trainer.train(data_processor, train_dataframes)

    utils.stop_stopwatch_and_print_running_time()

if __name__ == '__main__':
    main()
