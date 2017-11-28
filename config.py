import datetime
import os
import numpy as np
import operator
import time
import yaml
from random import seed
from utils import Utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

np.set_printoptions(linewidth=200, precision=2)
np.seterr(divide='ignore', invalid='ignore')

class ModelConfiguration(object):
    __slots__ = ('train_shuffle_examples_in_batches', 'train_schedule',
                 'train_shuffle_sentences', 'train_epochs', 'train_loss_optimizer', 'train_loss_optimizer_kwargs',
                 'train_continue_previous', 'train_add_summary_modulo', 'train_validation_modulo', 'train_validation_add_summary_modulo',
                 'train_rebuild_vocabulary_file', 'train_starter_learning_rate', 'train_visualization',

                 'data_random_seed', 'data_example_resolution', 'data_vocabulary_file',
                 'data_train_ratio', 'data_validation_ratio', 'data_sentences_to_read_num',
                 'data_train_matrices', 'data_train_dataset', 'data_batch_size',

                 'network_max_source_sequence_length', 'network_max_target_sequence_length',
                 'network_dropout_keep_probability', 'network_activation',
                 'network_hidden_layer_count', 'network_hidden_layer_cells', 'network_hidden_layer_cell_type',
                 'network_embedding_size', 'network_window_length', 'network_max_gradient_norm',

                 'inference_transducer_path',
                 'inference_decoder_type',
                 'inference_beam_width',
                 'use_train_model',

                 'model_directory', 'model_name',

                 'marker_padding', 'marker_analysis_divider', 'marker_start_of_sentence',
                 'marker_end_of_sentence', 'marker_unknown', 'marker_go',

                 'data_label_metadata_file')

    def __init__(self, parser):
        args = parser.parse_args()

        # Overridable optional variables
        self.train_loss_optimizer_kwargs = {}
        self.train_visualization = False
        self.data_sentences_to_read_num = None

        # For resolving absolute paths
        base_path = os.path.dirname(__file__)

        # If it is a new model
        if args.model_directory is None:
            if args.default_config is None:
                raise ValueError('You must provide the default config path a train a new model.')

            # Loading default config file
            with open(args.default_config, 'r', encoding='utf8') as default_config_file:
                default_config = yaml.safe_load(default_config_file)

            # Building model name
            new_dir_name = datetime.datetime.fromtimestamp(time.time()).strftime('%m%d-%H%M%S')+'.'+\
                           default_config['network']['hidden_layer_cell_type']+'x'+\
                           str(default_config['network']['hidden_layer_count'])+'x'+\
                           str(default_config['network']['hidden_layer_cells'])+'.'+\
                           default_config['train']['loss_optimizer'][:-9]+'.'+\
                           default_config['data']['example_resolution'][:4]

            self.model_directory = os.path.join(base_path, 'saved_models', new_dir_name)

            os.makedirs(self.model_directory)

            # Copy default config
            with open(os.path.join(self.model_directory, 'model_configuration.yaml'), 'w', encoding='utf8') as model_configuration_file:
                yaml.dump(default_config, model_configuration_file, default_flow_style=False)

            model_configuration = default_config

            self.train_continue_previous = False
        else:
            # Already existing model
            self.model_directory = args.model_directory
            if os.path.exists(self.model_directory) and os.path.isdir(self.model_directory):

                # Opening existing model config and merge onto the default config
                with open(os.path.join(self.model_directory, 'model_configuration.yaml'), 'r', encoding='utf8') as model_configuration_file:
                    model_configuration = yaml.safe_load(model_configuration_file)
                    #model_configuration = {**default_config, **model_configuration}
                self.train_continue_previous = True

            else:
                raise ValueError('Model does not exist:', args.model_directory)

        # Setting parameters
        for param_group_key, param_group in model_configuration.items():
            for param, value in param_group.items():
                setattr(self, param_group_key+'_'+param, value)

        # Setting non-configurable parameters
        self.model_name = 'saved_model.ckpt'
        self.marker_padding = 0
        self.marker_analysis_divider = 1
        self.marker_start_of_sentence = 2
        self.marker_end_of_sentence = 3
        self.marker_unknown = 4
        self.marker_go = 5

        # By default the program uses the latest model saved by validation cycle
        if hasattr(args, 'use_train_model'):
            self.use_train_model = args.use_train_model
        else:
            self.use_train_model = False

        # Building file paths
        self.data_label_metadata_file = os.path.join(base_path, 'data', 'metadata_'+self.data_example_resolution+'.tsv')
        self.data_train_matrices = os.path.join(base_path, model_configuration['data']['train_matrices'], self.data_example_resolution)
        self.data_train_dataset = os.path.join(base_path, model_configuration['data']['train_dataset'])
        self.data_vocabulary_file = os.path.join(base_path, 'data', 'vocabulary_' + self.data_example_resolution + '.tsv')

        # Analyses preprocessor will know if it has to.
        self.train_rebuild_vocabulary_file = False

        # Begin configuration
        if self.data_random_seed is not None:
            seed(self.data_random_seed)

    def printConfig(self):
        print('%90s' % ('Global configurations'))
        print("%-50s\t%s" % ('Key', 'Value'))
        print('-'*100)
        for k, v in sorted(Utils.fullvars(self).items(), key=operator.itemgetter(0)):
            print("%-50s\t%s" % (k, v))
        print('-' * 100)

