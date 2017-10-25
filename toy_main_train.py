from config.config import ModelConfiguration
from data_processing.toy_data_processor import ToyDataProcessor
from model.model_train import BuildTrainModel
from train import MorphDisamTrainer
from utils import Utils


def main():
    utils = Utils()
    utils.start_stopwatch()
    utils.redirect_stdout('toy-main-train')

    model_configuration = ModelConfiguration()

    model_configuration.max_source_sequence_length = 8
    model_configuration.max_target_sequence_length = 14

    # Loading train data
    train_data_processor = ToyDataProcessor(model_configuration)

    source_input_examples, target_input_examples, target_output_examples = train_data_processor.get_batches()

    # Building graph
    build_train_model = BuildTrainModel(model_configuration,
                                        train_data_processor.vocabulary,
                                        train_data_processor.inverse_vocabulary)

    train_graph = build_train_model.get_train_graph()

    model = build_train_model.create_model()

    # Begin training
    morph_disam_trainer = MorphDisamTrainer(train_graph, model_configuration, model)

    morph_disam_trainer.train(source_input_examples, target_input_examples, target_output_examples)

    utils.stop_stopwatch_and_print_running_time()

if __name__ == '__main__':
    main()
