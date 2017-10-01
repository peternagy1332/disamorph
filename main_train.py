from config import ModelConfiguration
from model_train import BuildTrainModel
from train import MorphDisamTrainer
from train_data_processor import TrainDataProcessor


def main():
    model_configuration = ModelConfiguration()

    # Loading train data
    train_data_processor = TrainDataProcessor(model_configuration)
    dataset, max_source_sequence_length, max_target_sequence_length = train_data_processor.process_dataset()

    train_data_processor.save_dataset_metadata()

    # Building graph
    morph_disam_model = BuildTrainModel(model_configuration,
                                        train_data_processor.vocabulary,
                                        max_source_sequence_length,
                                        max_target_sequence_length)

    model = morph_disam_model.create_model()

    # Begin training
    morph_disam_trainer = MorphDisamTrainer(model_configuration, model)

    morph_disam_trainer.train(dataset)


if __name__ == '__main__':
    main()
