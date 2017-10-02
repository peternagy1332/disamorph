from config import ModelConfiguration
from model_inference import BuildInferenceModel
from model_train import BuildTrainModel
from train_data_processor import TrainDataProcessor
import numpy as np


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

    to_predict = np.matrix([30943, 111, 94, 1, 15481, 18, 115, 1, 49157, 1, 5902, 70, 1, 45142, 18, 115, 1, 5902, 3, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32)

    print(build_inference_model.infer(infer_model, to_predict))


if __name__ == '__main__':
    main()