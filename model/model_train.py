from collections import namedtuple
import tensorflow as tf
from tensorflow.python.layers import core as layers_core


class BuildTrainModel(object):
    def __init__(self, model_configuration, vocabulary, inverse_vocabulary):
        self.__config = model_configuration
        self.__vocabulary = vocabulary
        self.__inverse_vocabulary = inverse_vocabulary

        self.__train_graph = tf.Graph()

    def get_train_graph(self):
        return self.__train_graph

    def create_model(self):
        with self.__train_graph.as_default():
            placeholders = self.__create_placeholders()

            embedding_matrix = self.create_embedding()

            encoder_outputs, encoder_state = self.create_encoder(embedding_matrix, placeholders.encoder_inputs)

            logits, output_sequences = self.__create_decoder(embedding_matrix, encoder_outputs, encoder_state, placeholders)

            Model = namedtuple('Model', ['placeholders', 'logits', 'output_sequences'])

            return Model(
                placeholders=placeholders,
                logits=logits,
                output_sequences=output_sequences
            )

    def create_embedding(self):
        with tf.variable_scope('embedding'):
            embedding_matrix = tf.get_variable('embedding_matrix',
                                               [len(self.__inverse_vocabulary), self.__config.embedding_size],
                                               dtype=tf.float32)
            return embedding_matrix


    def create_rnn(self, half=False):
        cells = []

        layers = self.__config.hidden_layer_count if half==False else self.__config.hidden_layer_count // 2

        for i in range(layers):
            if self.__config.hidden_layer_cell_type == 'LSTM':
                cells.append(tf.nn.rnn_cell.BasicLSTMCell(self.__config.hidden_layer_cells))
            elif self.__config.hidden_layer_cell_type == 'GRU':
                cells.append(tf.nn.rnn_cell.GRUCell(self.__config.hidden_layer_cells))

        return tf.nn.rnn_cell.MultiRNNCell(cells)

    def __create_placeholders(self):
        Placeholders = namedtuple('Placeholders', ['encoder_inputs', 'decoder_inputs', 'decoder_outputs'])

        with tf.variable_scope('placeholders'):
            encoder_inputs = tf.placeholder(tf.int32,
                                            [None,
                                             self.__config.max_source_sequence_length],
                                            'encoder_inputs')

            decoder_inputs = tf.placeholder(tf.int32,
                                            [None,
                                             self.__config.max_target_sequence_length],
                                            'decoder_inputs')

            decoder_outputs = tf.placeholder(tf.int32,
                                             [None,
                                              self.__config.max_target_sequence_length],
                                             'decoder_outputs')

            return Placeholders(
                encoder_inputs=encoder_inputs,
                decoder_inputs=decoder_inputs,
                decoder_outputs=decoder_outputs
            )

    def create_encoder(self, embedding_matrix, encoder_inputs):
        with tf.variable_scope('encoder'):

            forward_cells = self.create_rnn(True)
            backward_cells = self.create_rnn(True)

            embedding_input = tf.nn.embedding_lookup(embedding_matrix, encoder_inputs)

            o, s = tf.nn.bidirectional_dynamic_rnn(
                forward_cells, backward_cells, embedding_input, dtype=tf.float32,
                sequence_length=tf.count_nonzero(encoder_inputs, axis=1, dtype=tf.int32),
                time_major=False
            )

            encoder_outputs = tf.concat(o, -1)
            encoder_state = []
            for i in range(self.__config.hidden_layer_count // 2):
                encoder_state.append(s[0][i])
                encoder_state.append(s[1][i])
            encoder_state = tuple(encoder_state)

            # encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
            #     encoder_rnn, embedding_input,
            #     sequence_length=tf.count_nonzero(encoder_inputs, axis=1, dtype=tf.int32),
            #     dtype=tf.float32,
            #     time_major=False
            # )

            return encoder_outputs, encoder_state

    def __create_decoder(self, embedding_matrix, encoder_outputs, encoder_state, placeholders):
        with tf.variable_scope('decoder'):
            decoder_rnn = self.create_rnn(False)

            decoder_inputs_sequence_length = tf.count_nonzero(placeholders.decoder_inputs, axis=1, dtype=tf.int32)

            mechanism = tf.contrib.seq2seq.LuongAttention(
                self.__config.hidden_layer_cells, encoder_outputs,
                memory_sequence_length=decoder_inputs_sequence_length,
                scale=False
            )

            decoder_rnn = tf.contrib.seq2seq.AttentionWrapper(
                decoder_rnn, mechanism, attention_layer_size=self.__config.hidden_layer_cells
            )

            embedding_input = tf.nn.embedding_lookup(embedding_matrix, placeholders.decoder_inputs)

            helper = tf.contrib.seq2seq.TrainingHelper(embedding_input, decoder_inputs_sequence_length, time_major=False)

            projection_layer = layers_core.Dense(len(self.__inverse_vocabulary), use_bias=False)

            decoder_initial_state = decoder_rnn.zero_state(self.__config.batch_size, tf.float32).clone(cell_state=encoder_state)

            decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=decoder_rnn,
                helper=helper,
                initial_state=decoder_initial_state,
                output_layer=projection_layer
            )

            # output_time_major=True
            final_outputs, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
                decoder #, output_time_major=False, swap_memory=True
            )

            logits = final_outputs.rnn_output
            output_sequences = final_outputs.sample_id

            # label_first_dim(self.batch_size * self.max_target_sequence_length) = logit_first_dim(self.batch_size * self.max_target_Sequence_length)
            #vertical_padding = tf.zeros([self.__config.max_target_sequence_length-tf.reduce_max(decoder_inputs_sequence_length), len(self.__inverse_vocabulary)], dtype=tf.float32)
            #logits = tf.map_fn(lambda logit: tf.concat([logit, vertical_padding], axis=0), logits)

            return logits, output_sequences
