import datetime
import os

import numpy as np
import pandas as pd
import operator
import argparse

# Silence TensorFlow
import time

import yaml

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# GPU ID
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from utils import Utils

np.set_printoptions(linewidth=200, precision=2)

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

class ModelConfiguration(object):
    __slots__ = ('train_vocabulary_cache', 'train_files_corpus', 'train_save_modulo', 'train_matrices',
                 'train_early_stop_after_not_decreasing_loss_num', 'train_shuffle_sentences', 'train_epochs', 'train_loss_optimizer', 'train_loss_optimizer_kwargs',
                 'train_decay_rate', 'train_decay_steps', 'train_continue_previous', 'train_decaying_learning_rate', 'train_decay_type', 'train_shuffle_examples_in_batches',
                 'embedding_labels_metadata', 'network_dropout_keep_probability', 'network_activation',
                 'model_directory', 'model_name',

                 'hidden_layer_count', 'hidden_layer_cells', 'hidden_layer_cell_type', 'train_rebuild_vocabulary_file',
                 'test_sentences_rate', 'max_source_sequence_length', 'max_target_sequence_length', 'embedding_size', 'batch_size',
                 'window_length', 'vocabulary_start_index', 'rows_to_read_num', 'max_gradient_norm', 'train_starter_learning_rate',

                 'marker_padding', 'marker_analysis_divider', 'marker_start_of_sentence', 'marker_end_of_sentence', 'marker_unknown', 'marker_go',

                 'data_random_seed',
                 'max_source_sequence_length', 'max_target_sequence_length', 'inference_batch_size', 'inference_maximum_iterations', 'analyses_path',
                 'transducer_path')

    def __init__(self, parser):

        base_path = os.path.dirname(__file__)

        settings = {
            'default_args': {
                'model_directory': os.path.join(base_path, 'logs', 'model-'+datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H:%M:%S')),
                'default_config': os.path.join(base_path, 'configs','default_config.yaml')
            },
            'model_name': 'saved_model.ckpt',
            'config_name': 'model_configuration.yaml'
        }

        args = parser.parse_args()

        if args.default_config is None:
            args.default_config = settings['default_args']['default_config']

        if args.model_directory is None:
            args.model_directory = settings['default_args']['model_directory']

        with open(args.default_config, 'r', encoding='utf8') as default_config_file:
            default_config = yaml.safe_load(default_config_file)

        # Already existing model
        config_file_path = os.path.join(args.model_directory, settings['config_name'])
        if os.path.exists(args.model_directory) and os.path.isdir(args.model_directory):
            with open(config_file_path, 'r', encoding='utf8') as model_configuration_file:
                model_configuration = yaml.safe_load(model_configuration_file)
                model_configuration = {**default_config, **model_configuration}
            self.train_continue_previous = True
        else:
            # New model
            os.makedirs(args.model_directory)
            with open(config_file_path, 'w', encoding='utf8') as model_configuration_file:
                yaml.dump(default_config, model_configuration_file, default_flow_style=False)
            model_configuration = default_config
            self.train_continue_previous = False

        self.model_directory = args.model_directory
        self.model_name = settings['model_name']

        self.embedding_size = model_configuration['network']['embedding_size']
        self.batch_size = model_configuration['train']['batch_size']
        self.window_length = model_configuration['network']['window_length']

        self.marker_padding = 0
        self.marker_analysis_divider = 1
        self.marker_start_of_sentence = 2
        self.marker_end_of_sentence = 3
        self.marker_unknown = 4
        self.marker_go = 5
        self.vocabulary_start_index = 6

        self.embedding_labels_metadata = os.path.join(base_path,'data','embedding_labels.tsv')

        if 'dropout_keep_probability' in model_configuration['network'].keys(): self.network_dropout_keep_probability = model_configuration['network']['dropout_keep_probability']
        else: self.network_dropout_keep_probability = None

        if 'activation' in model_configuration['network'].keys(): self.network_activation = model_configuration['network']['activation']
        else: self.network_activation = None

        if 'random_seed' in model_configuration['data'].keys(): self.data_random_seed = model_configuration['data']['random_seed']
        else: self.data_random_seed = None

        self.rows_to_read_num = model_configuration['data']['rows_to_read_num']
        self.max_gradient_norm = model_configuration['network']['max_gradient_norm']
        self.hidden_layer_count = model_configuration['network']['hidden_layer_count']
        self.hidden_layer_cells = model_configuration['network']['hidden_layer_cells']
        self.hidden_layer_cell_type = model_configuration['network']['hidden_layer_cell_type']
        self.train_loss_optimizer = model_configuration['train']['loss_optimizer']

        if 'loss_optimizer_kwargs' in model_configuration['train']:self.train_loss_optimizer_kwargs = model_configuration['train']['loss_optimizer_kwargs']
        else: self.train_loss_optimizer_kwargs = {}

        if 'shuffle_examples_in_batches' in model_configuration['train']:self.train_shuffle_examples_in_batches = model_configuration['train']['shuffle_examples_in_batches']
        else: self.train_shuffle_examples_in_batches = False

        self.train_decay_type = model_configuration['train']['decay_type']
        self.train_decaying_learning_rate = model_configuration['train']['decaying_learning_rate']
        self.train_starter_learning_rate = model_configuration['train']['starter_learning_rate']
        self.train_decay_steps = model_configuration['train']['decay_steps']
        self.train_decay_rate = model_configuration['train']['decay_rate']

        self.train_matrices = os.path.join(base_path, model_configuration['data']['train_matrices'])
        self.train_rebuild_vocabulary_file = model_configuration['data']['rebuild_vocabulary_file']
        self.train_epochs = model_configuration['train']['epochs']
        self.train_vocabulary_cache = os.path.join(base_path,model_configuration['data']['vocabulary_cache'])
        self.train_files_corpus = os.path.join(base_path, model_configuration['data']['train_dataset'])
        self.train_early_stop_after_not_decreasing_loss_num = model_configuration['train']['early_stop_after_not_decreasing_loss_num']
        self.train_save_modulo = model_configuration['train']['save_modulo']
        self.train_shuffle_sentences = model_configuration['train']['shuffle_sentences']
        self.test_sentences_rate = model_configuration['test']['sentences_rate']

        self.inference_batch_size = model_configuration['inference']['batch_size']
        self.inference_maximum_iterations = model_configuration['inference']['maximum_iterations']

        self.max_source_sequence_length = model_configuration['network']['max_source_sequence_length']
        self.max_target_sequence_length = model_configuration['network']['max_target_sequence_length']

        self.transducer_path = model_configuration['inference']['transducer_path']

    def printConfig(self):
        print('%90s' % ('Global configurations'))
        print("%80s | %s" % ('Value', 'Key'))
        print('-'*160)
        for k, v in sorted(Utils.fullvars(self).items(), key=operator.itemgetter(0)):
            print("%80s : %s" % (v, k))
