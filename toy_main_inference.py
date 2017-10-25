from data_processing.toy_data_processor import ToyDataProcessor
from model.model_inference import BuildInferenceModel
from model.model_train import BuildTrainModel
from toy_inference import input_to_output
from config.config import ModelConfiguration
from utils import Utils


def main():
    utils = Utils()
    utils.start_stopwatch()
    utils.redirect_stdout('toy-main-inference')

    model_configuration = ModelConfiguration()
    model_configuration.max_source_sequence_length = 8
    model_configuration.max_target_sequence_length = 14
    model_configuration.inference_batch_size = 1

    # Loading train data
    data_processor = ToyDataProcessor(model_configuration)

    # Building train model
    build_train_model = BuildTrainModel(model_configuration,
                                        data_processor.vocabulary,
                                        data_processor.inverse_vocabulary)

    build_train_model.create_model()

    # Building inference model
    build_inference_model = BuildInferenceModel(model_configuration,
                                                data_processor.inverse_vocabulary,
                                                build_train_model)

    inference_model = build_inference_model.create_model()

    while True:
        print('Enter a word: ')
        text = input()
        input_to_output(inference_model, text, model_configuration, data_processor.vocabulary, data_processor.inverse_vocabulary)
        if text is None: break

    utils.stop_stopwatch_and_print_running_time()

if __name__ == '__main__':
    main()