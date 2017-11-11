from config import ModelConfiguration
from data_processing.toy_data_processor import ToyDataProcessor
from toy_inferencer import ToyInferencer
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

    inferencer = ToyInferencer(model_configuration, data_processor)

    while True:
        print('Enter a word: ', end='')
        text = input()
        print(inferencer.inference_batch_to_output(text))
        if text is None: break

    utils.stop_stopwatch_and_print_running_time()

if __name__ == '__main__':
    main()