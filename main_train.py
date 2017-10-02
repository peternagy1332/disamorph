from config import ModelConfiguration
from model_train import BuildTrainModel
from train import MorphDisamTrainer
from train_data_processor import TrainDataProcessor


def main():
    model_configuration = ModelConfiguration()

    # Loading train data
    train_data_processor = TrainDataProcessor(model_configuration)
    dataset, dataset_metadata = train_data_processor.process_dataset()

    train_data_processor.save_dataset_metadata()

    # Building graph
    build_train_model = BuildTrainModel(model_configuration,
                                        train_data_processor.vocabulary,
                                        train_data_processor.inverse_vocabulary,
                                        dataset_metadata)

    train_graph = build_train_model.get_train_graph()

    model = build_train_model.create_model()

    # Begin training
    morph_disam_trainer = MorphDisamTrainer(train_graph, model_configuration, model)

    morph_disam_trainer.train(dataset)


if __name__ == '__main__':
    main()