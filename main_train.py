from random import shuffle

from config.config import ModelConfiguration
from data_processing.data_processor import DataProcessor
from disambiguator import Disambiguator
from model.model_train import BuildTrainModel
from train import MorphDisamTrainer
from utils import Utils


def main():
    utils = Utils()
    utils.redirect_stdout('main-train')
    model_configuration = ModelConfiguration()

    # Loading train data
    train_data_processor = DataProcessor(model_configuration)

    sentence_daraframes = train_data_processor.get_sentence_dataframes()

    test_dataframe_id = int(round(len(sentence_daraframes)*model_configuration.test_sentences_rate))
    test_dataframes = sentence_daraframes[-test_dataframe_id:]

    train_dataframes = sentence_daraframes[:-test_dataframe_id]
    if model_configuration.train_shuffle_sentences:
        shuffle(train_dataframes)

    train_batches = train_data_processor.train_dataframes_to_batches(train_dataframes)

    # Building graph
    build_train_model = BuildTrainModel(model_configuration,
                                        train_data_processor.vocabulary,
                                        train_data_processor.inverse_vocabulary)

    train_graph = build_train_model.get_train_graph()

    model = build_train_model.create_model()

    # Begin training
    morph_disam_trainer = MorphDisamTrainer(train_graph, model_configuration, model)

    #morph_disam_trainer.train(train_batches)

    # Evaluating model
    disambiguator = Disambiguator(model_configuration)

    print('Evaluating model on train dataset...')
    morph_disam_trainer.evaluate_model(disambiguator, train_dataframes, 10)

    print('Evaluating model on test dataset...')
    morph_disam_trainer.evaluate_model(disambiguator, test_dataframes)

if __name__ == '__main__':
    main()
