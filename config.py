import os


class ModelConfiguration(object):
    __slots__ = ('train_files_tags',
                 'train_files_roots',
                 'train_files_corpus',
                 'train_files_losses',
                 'train_files_dataset_metadata',
                 'train_files_save_model',
                 'train_save_modulo',
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

    def __init__(self):
        self.embedding_size = 10
        self.num_cells = 32
        self.batch_size = 6
        self.window_length = 5
        self.marker_padding = 0
        self.marker_analysis_divider = 1
        self.marker_start_of_sentence = 2
        self.marker_end_of_sentence = 3
        self.marker_unknown = 4
        self.vocabulary_start_index = 5
        self.nrows = 2
        self.max_gradient_norm = 1  # 1..5
        self.learning_rate = 1
        self.train_epochs = 20
        self.train_files_tags = os.path.join('data', 'tags.txt')
        self.train_files_roots = os.path.join('data', 'roots.txt')
        self.train_files_corpus = os.path.join('data', 'szeged', '*')
        self.train_files_losses = os.path.join('logs', 'losses.txt')
        self.train_files_dataset_metadata = os.path.join('logs', 'dataset_metadata.txt')
        self.train_files_save_model = os.path.join('logs', 'model', 'saved_model.ckpt')
        self.train_save_modulo = 10