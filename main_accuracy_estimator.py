import argparse
import numpy as np
import os

from config import ModelConfiguration
from data_processing.analyses_processor import AnalysesProcessor
from data_processing.data_processor import DataProcessor
from disambiguator import Disambiguator
from utils import Utils

def estimate_accuracies_on_dataset(model_configuration, disambiguator, dataset, data_processor):
    print('\tDataset stats:')
    print('\t\tsource_input_examples:', dataset.source_input_examples.shape)
    print('\t\ttarget_input_examples:', dataset.target_input_examples.shape)
    print('\t\ttarget_output_examples:', dataset.target_output_examples.shape)
    print_analyses = True

    batch_accuracies = []
    try:
        print('\t%-20s| %s' % ("Batch", "Accuracy"))
        batch_pointer = 0
        while batch_pointer + model_configuration.data_batch_size < dataset.source_input_examples.shape[0]:
            source_input_matrix = dataset.source_input_examples[batch_pointer: batch_pointer + model_configuration.data_batch_size]
            target_output_matrix = dataset.target_output_examples[batch_pointer:batch_pointer + model_configuration.data_batch_size]

            scores, output_sequences = disambiguator.feed_into_network(source_input_matrix)

            target_sequence_lengths = np.count_nonzero(target_output_matrix, axis=1)

            sequence_level_accuracy = []
            for rowid, target_sequence_length in enumerate(target_sequence_lengths.tolist()):
                output_sequence = output_sequences[rowid][:target_sequence_length]
                target_output_sequence = target_output_matrix[rowid][:target_sequence_length]
                if print_analyses:
                    print('\t%-60s| %s' % ("".join(disambiguator.analyses_processor.lookup_ids_to_features(target_output_sequence)).replace("<PAD>", ""),
                                           "".join(disambiguator.analyses_processor.lookup_ids_to_features(output_sequence)).replace("<PAD>", "")))
                sequence_level_accuracy.append(np.mean(np.equal(output_sequence, target_output_sequence)))

            batch_accuracy = sum(sequence_level_accuracy) * 100 / len(sequence_level_accuracy)
            batch_accuracies.append(batch_accuracy)
            print('\t%-20d| %g' % (batch_pointer/model_configuration.data_batch_size, batch_accuracy))

            batch_pointer += model_configuration.data_batch_size
    except KeyboardInterrupt:
        pass
    print('\tTotal accuracies: min - %-15g max - %-15g avg - %-15g' % (min(batch_accuracies), max(batch_accuracies), (sum(batch_accuracies)/len(batch_accuracies))))

def main():
    parser = argparse.ArgumentParser(description='Hungarian morphological disambiguator')
    parser.add_argument('-dcfg', '--default-config', default=os.path.join('default_configs', 'character.yaml'))
    parser.add_argument('-m', '--model-directory', required=True)
    parser.add_argument('-t', '--use-train-model', default=False, action='store_true')

    model_configuration = ModelConfiguration(parser)
    model_configuration.train_shuffle_examples_in_batches = False
    model_configuration.train_shuffle_sentences = False

    utils = Utils(model_configuration)
    utils.start_stopwatch()
    utils.redirect_stdout('main-evaluation')

    print('Initializing data utilities...')
    analyses_processor = AnalysesProcessor(model_configuration)
    data_processor = DataProcessor(model_configuration, analyses_processor)
    utils.print_elapsed_time()

    print('Reading sentence dicts...')
    sentence_dicts = data_processor.get_sentence_dicts(True)
    utils.print_elapsed_time()

    print('Getting example matrices...')
    train_dataset, validation_dataset, test_dataset = data_processor.get_example_matrices(sentence_dicts)
    utils.print_elapsed_time()

    print('Initializing disambiguator instance...')
    disambiguator = Disambiguator(model_configuration, analyses_processor)
    utils.print_elapsed_time()
    print()

    print('Estimating accuracies on train dataset...')
    estimate_accuracies_on_dataset(model_configuration, disambiguator, train_dataset, data_processor)
    utils.print_elapsed_time()
    print()

    print('Estimating accuracies on validation dataset...')
    estimate_accuracies_on_dataset(model_configuration, disambiguator, validation_dataset, data_processor)
    utils.print_elapsed_time()
    print()

    print('Estimating accuracies on test dataset...')
    estimate_accuracies_on_dataset(model_configuration, disambiguator, test_dataset, data_processor)
    utils.print_elapsed_time()

if __name__ == '__main__':
    main()