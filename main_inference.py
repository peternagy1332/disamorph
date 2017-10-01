from config import ModelConfiguration
from inference import MorphDisamInference
from model import MorphDisamModel
from train_data_processor import TrainDataProcessor
import numpy as np


def main():
    model_configuration = ModelConfiguration()

    # Loading train data meta
    train_data_processor = TrainDataProcessor(model_configuration)

    dataset_metadata = train_data_processor.load_dataset_metadata()

    print('Dataset metadata: ', dataset_metadata)

    # Building graph
    morph_disam_model = MorphDisamModel(model_configuration,
                                        train_data_processor.vocabulary,
                                        dataset_metadata.max_source_sequence_length,
                                        dataset_metadata.max_target_sequence_length)

    model = morph_disam_model.create_model()

    # Begin training
    morph_disam_inference = MorphDisamInference(model_configuration, model)

    to_predict = np.ndarray([30943, 111, 94, 1, 15481, 18, 115, 1, 49157, 1, 5902, 70, 1, 45142, 18, 115, 1, 5902, 3, 0, 0, 0, 0])

    print(morph_disam_inference.infer(to_predict))


if __name__ == '__main__':
    main()