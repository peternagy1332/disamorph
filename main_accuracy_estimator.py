import numpy as np
import argparse
import os
from disamorph import ModelConfiguration, Utils
from disamorph.data_processing.analyses_processor import AnalysesProcessor
from disamorph.data_processing.data_processor import DataProcessor
from disamorph.disambiguator import Disambiguator


def estimate_accuracies_on_dataset(model_configuration, disambiguator, dataset):
    print('\tDataset stats:')
    print('\t\tsource_input_examples:', dataset.source_input_examples.shape)
    print('\t\ttarget_input_examples:', dataset.target_input_examples.shape)
    print('\t\ttarget_output_examples:', dataset.target_output_examples.shape)
    print_analyses = False

    batch_accuracies = []
    failed_feature_to_count = dict()
    try:
        if not print_analyses:
            print('\t%-20s| %s' % ("Batch", "Accuracy"))

        batch_pointer = 0
        while batch_pointer + model_configuration.data_batch_size <= dataset.source_input_examples.shape[0]:
            if print_analyses:
                print('-'*100)
                print('Batch: ', batch_pointer/model_configuration.data_batch_size)

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

                if output_sequence.shape[0]<target_sequence_length:
                    output_sequence = np.concatenate((output_sequence, np.zeros((target_sequence_length-output_sequence.shape[0],))),axis=0)

                for t, o in zip(target_output_sequence.tolist(), output_sequence.tolist()):
                    if t != o:
                        if t in failed_feature_to_count.keys():
                            failed_feature_to_count[t]+=1
                        else:
                            failed_feature_to_count.setdefault(t, 1)

                sequence_level_accuracy.append(np.mean(np.equal(target_output_sequence, output_sequence)))

            batch_accuracies.append(sum(sequence_level_accuracy) * 100 / len(sequence_level_accuracy))
            if print_analyses:
                print('\tBatch accuracy:', batch_accuracies[-1])
            else:
                print('\t%-20d| %g' % (batch_pointer/model_configuration.data_batch_size, batch_accuracies[-1]))

            batch_pointer += model_configuration.data_batch_size
    except KeyboardInterrupt:
        pass

    if model_configuration.data_batch_size > dataset.source_input_examples.shape[0]:
        print('Dataset is too small (batch padding is not avaiable in accuracy estimator)!')
    else:
        print('Writing output CSV...')
        with open('accuracy_estimator_output_'+model_configuration.data_example_resolution+'.csv', 'w') as f:
            f.writelines(list(map(lambda l: str(l) + os.linesep, batch_accuracies)))

        with open('failed_feature_to_count_'+model_configuration.data_example_resolution+'.csv', 'w', encoding='utf-8') as f:
            f.writelines(map(lambda kv: str(disambiguator.analyses_processor.lookup_ids_to_features([kv[0]])[0])+'\t'+str(kv[1])+os.linesep,zip(failed_feature_to_count.keys(), failed_feature_to_count.values())))

        if len(batch_accuracies) > 0:
            print('\tTotal accuracies: min - %-15g max - %-15g avg - %-15g' % (min(batch_accuracies), max(batch_accuracies), (sum(batch_accuracies)/len(batch_accuracies))))

def main():
    parser = argparse.ArgumentParser(description='Disamorph: A Hungarian morphological disambiguator using sequence-to-sequence neural networks.')
    parser.add_argument('-m', '--model-directory', required=True, help='Path to the model directory.')
    parser.add_argument('-t', '--use-train-model', default=False, action='store_true', help='Whether to use the train instead of the validation model.')
    args = parser.parse_args()

    model_configuration = ModelConfiguration(args)
    model_configuration.train_shuffle_examples_in_batches = False
    model_configuration.train_shuffle_sentences = False

    utils = Utils(model_configuration)
    utils.start_stopwatch()
    #utils.redirect_stdout('main-evaluation')

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
    print('=' * 100)

    # print('Estimating accuracies on train dataset...')
    # estimate_accuracies_on_dataset(model_configuration, disambiguator, train_dataset)
    # utils.print_elapsed_time()
    # print('='*100)
    #
    # print('Estimating accuracies on validation dataset...')
    # estimate_accuracies_on_dataset(model_configuration, disambiguator, validation_dataset)
    # utils.print_elapsed_time()
    # print('=' * 100)

    print('Estimating accuracies on test dataset...')
    estimate_accuracies_on_dataset(model_configuration, disambiguator, test_dataset)
    utils.print_elapsed_time()

if __name__ == '__main__':
    main()