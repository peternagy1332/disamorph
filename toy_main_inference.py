import numpy as np
from model.model_inference import BuildInferenceModel

from config.config import ModelConfiguration
from data_processing.toy_train_data_processor import TrainDataProcessor
from model.model_train import BuildTrainModel

np.set_printoptions(linewidth=200, precision=2)

def main():
    model_configuration = ModelConfiguration()

    # Loading train data meta
    train_data_processor = TrainDataProcessor(model_configuration)

    dataset_metadata = train_data_processor.load_dataset_metadata()

    # Building graph
    build_train_model = BuildTrainModel(model_configuration,
                                        train_data_processor.vocabulary,
                                        train_data_processor.inverse_vocabulary,
                                        dataset_metadata)

    train_model = build_train_model.create_model()

    # Begin training
    build_inference_model = BuildInferenceModel(model_configuration,
                                                train_data_processor.inverse_vocabulary,
                                                dataset_metadata,
                                                build_train_model)

    infer_model = build_inference_model.create_model()

    to_predict = np.matrix([ 6,  3,  0,  0,  0,  0,  0], dtype=np.int32)

    print(build_inference_model.infer(infer_model, to_predict))


if __name__ == '__main__':
    main()