import os

from config import ModelConfiguration
from data_processing.toy_data_processor import ToyDataProcessor
from model.model_train import BuildTrainModel
from seq2seq_trainer import Seq2SeqTrainer
from toy_inferencer import ToyInferencer
from utils import Utils


def main():
    utils = Utils()
    utils.start_stopwatch()
    utils.redirect_stdout('toy-main-train')

    model_configuration = ModelConfiguration()

    model_configuration.max_source_sequence_length = 8
    model_configuration.max_target_sequence_length = 14
    model_configuration.train_epochs = 10000
    model_configuration.rows_to_read_num = None
    model_configuration.train_files_save_model = os.path.join('logs', 'model', 'toy_main_train.ckpt')
    model_configuration.train_continue_previous = True
    model_configuration.printConfig()

    # Loading train data
    data_processor = ToyDataProcessor(model_configuration)

    source_input_examples, target_input_examples, target_output_examples = data_processor.get_batches()

    model_configuration.inference_batch_size = model_configuration.batch_size = source_input_examples.shape[0]

    # Building graph
    build_train_model = BuildTrainModel(model_configuration, data_processor.vocabulary, data_processor.inverse_vocabulary)

    train_model = build_train_model.create_model()

    # Begin training
    seq2seq_trainer = Seq2SeqTrainer(build_train_model.train_graph, model_configuration, train_model)

    seq2seq_trainer.train(source_input_examples, target_input_examples, target_output_examples)

    inferencer = ToyInferencer(model_configuration, data_processor)

    inferencer.evaluate_model(source_input_examples, target_output_examples)

    utils.stop_stopwatch_and_print_running_time()

if __name__ == '__main__':
    main()
