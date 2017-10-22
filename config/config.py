import os
import numpy as np

from utils import Utils

np.set_printoptions(linewidth=200, precision=2)

class ModelConfiguration(object):
    __slots__ = ('train_files_tags',
                 'train_files_roots',
                 'train_files_corpus',
                 'train_files_losses',
                 'train_files_save_model',
                 'train_save_modulo',
                 'train_early_stop_after_not_decreasing_loss_num',
                 'train_shuffle_sentences',

                 'hidden_layer_count',
                 'hidden_layer_cells',
                 'hidden_layer_cell_type',

                 'test_sentences_rate',
                 'max_source_sequence_length',
                 'max_target_sequence_length',
                 'embedding_size',
                 'batch_size',
                 'window_length',
                 'marker_padding',
                 'marker_analysis_divider',
                 'marker_start_of_sentence',
                 'marker_end_of_sentence',
                 'marker_unknown',
                 'vocabulary_start_index',
                 'rows_to_read_num',
                 'max_gradient_norm',
                 'learning_rate',
                 'train_epochs',
                 'max_source_sequence_length',
                 'max_target_sequence_length',
                 'inference_batch_size',
                 'inference_maximum_iterations',
                 'analyses_path')

    def __init__(self):
        self.embedding_size = 256
        self.batch_size = 64
        self.window_length = 5

        self.marker_padding = 0
        self.marker_analysis_divider = 1
        self.marker_start_of_sentence = 2
        self.marker_end_of_sentence = 3
        self.marker_unknown = 4

        self.vocabulary_start_index = 5
        self.rows_to_read_num = 1000
        self.max_gradient_norm = 1  # 1..5
        self.learning_rate = 1

        self.train_epochs = 100
        self.train_files_tags = os.path.join('data', 'tags.txt')
        self.train_files_roots = os.path.join('data', 'roots.txt')
        self.train_files_corpus = os.path.join('data', 'szeged-judit', 'utas.conll-2009_ready.disamb.new')
        #self.train_files_corpus = os.path.join('data', 'szeged-judit', '*')
        self.train_files_losses = os.path.join('logs', 'losses.txt')
        self.train_files_save_model = os.path.join('logs', 'model', 'saved_model.ckpt')
        self.train_early_stop_after_not_decreasing_loss_num = 5
        self.train_save_modulo = 10
        self.train_shuffle_sentences = True
        self.test_sentences_rate = 0.1

        self.hidden_layer_count = 16
        self.hidden_layer_cells = 64
        self.hidden_layer_cell_type = 'GRU'

        self.inference_batch_size = 512
        self.inference_maximum_iterations = 10

        self.analyses_path = os.path.join('data', 'analyses.txt')
        self.max_source_sequence_length = 64
        self.max_target_sequence_length = 32

        print('CONFIGURATION')
        for k, v in Utils.fullvars(self).items():
            print('\t',k,'\t', v)
