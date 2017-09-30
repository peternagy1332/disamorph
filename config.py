class Dataset(object):
    __slots__ = ('source_input_batches',
                 'target_input_batches',
                 'target_output_batches')


class ModelConfiguration(object):
    __slots__ = ('train_files_tags',
                 'train_files_roots',
                 'train_files_corpus',
                 'embedding_size',
                 'num_cells',
                 'batch_size',
                 'window_length',
                 'marker_padding',
                 'marker_analysis_divider',
                 'marker_start_of_sentence',
                 'marker_end_of_sentence',
                 'marker_unknown',
                 'vocabulary_start_index',
                 'nrows',
                 'max_gradient_norm',
                 'learning_rate',
                 'train_epochs',
                 'max_source_sequence_length',
                 'max_target_sequence_length')
