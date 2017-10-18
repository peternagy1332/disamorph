import numpy as np
from model.model_inference import BuildInferenceModel

from config.config import ModelConfiguration
from data_processing.toy_train_data_processor import TrainDataProcessor
from model.model_train import BuildTrainModel


def main():
    model_configuration = ModelConfiguration()

    # Loading train data
    train_data_processor = TrainDataProcessor(model_configuration)

    # Building graph
    build_train_model = BuildTrainModel(model_configuration,
                                        train_data_processor.vocabulary,
                                        train_data_processor.inverse_vocabulary)

    train_model = build_train_model.create_model()

    # Begin training
    build_inference_model = BuildInferenceModel(model_configuration,
                                                train_data_processor.inverse_vocabulary,
                                                build_train_model)

    infer_model = build_inference_model.create_model()

    to_predict = np.matrix([ 6,  3,  0,  0,  0,  0,  0], dtype=np.int32)

    #print(build_inference_model.infer(infer_model, to_predict))


if __name__ == '__main__':
    main()