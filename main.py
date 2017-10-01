import os
from train_data_processor import TrainDataProcessor
from config import ModelConfiguration, Dataset
from model import MorphDisamModel
from train import MorphDisamTrainer

def main():
    # Should be loaded from YAML
    model_configuration = ModelConfiguration()
    model_configuration.embedding_size=10
    model_configuration.num_cells=32
    model_configuration.batch_size=6
    model_configuration.window_length=5
    model_configuration.marker_padding=0
    model_configuration.marker_analysis_divider=1
    model_configuration.marker_start_of_sentence=2
    model_configuration.marker_end_of_sentence=3
    model_configuration.marker_unknown=4
    model_configuration.vocabulary_start_index=5
    model_configuration.nrows=5
    model_configuration.max_gradient_norm=1  # 1..5
    model_configuration.learning_rate=1
    model_configuration.train_epochs=100
    model_configuration.train_files_tags=os.path.join('data', 'tags.txt')
    model_configuration.train_files_roots=os.path.join('data', 'roots.txt')
    model_configuration.train_files_corpus=os.path.join('data', 'szeged', '*')

    # Loading train data
    train_data_processor = TrainDataProcessor(model_configuration)
    dataset, max_source_sequence_length, max_target_sequence_length, vocabulary = train_data_processor.process_dataset()

    # Building graph
    morph_disam_model = MorphDisamModel(model_configuration,
                                            vocabulary,
                                            max_source_sequence_length,
                                            max_target_sequence_length)

    placeholders, logits = morph_disam_model.create_network()

    # Begin training
    morph_disam_trainer = MorphDisamTrainer(model_configuration)
    morph_disam_trainer.train(placeholders, logits, dataset)


if __name__ == '__main__':
    main()
