from collections import namedtuple
import tensorflow as tf
from tensorflow.python.layers import core as layers_core


class MorphDisamModel(object):
    def __init__(self, model_configuration, vocabulary, max_source_sequence_length, max_target_sequence_length):
        self.__config = model_configuration

        self.__vocabulary = vocabulary
        self.__max_source_sequence_length = max_source_sequence_length
        self.__max_target_sequence_length = max_target_sequence_length

    def __create_placeholders(self):
        Placeholders = namedtuple('Placeholders', ['encoder_inputs', 'decoder_inputs', 'decoder_outputs'])

        with tf.variable_scope('placeholders'):

            encoder_inputs = tf.placeholder(tf.int32,
                                            [self.__config.batch_size, self.__max_source_sequence_length],
                                            'encoder_inputs')

            decoder_inputs = tf.placeholder(tf.int32,
                                            [self.__config.batch_size, self.__max_target_sequence_length],
                                            'decoder_inputs')

            decoder_outputs = tf.placeholder(tf.int32,
                                             [self.__config.batch_size, self.__max_target_sequence_length],
                                             'decoder_outputs')

            return Placeholders(
                encoder_inputs=encoder_inputs,
                decoder_inputs=decoder_inputs,
                decoder_outputs=decoder_outputs
            )

    def __create_encoder(self, embedding_matrix, encoder_inputs):
        with tf.variable_scope('encoder'):
            encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.__config.num_cells)

            embedding_input = tf.nn.embedding_lookup(embedding_matrix, encoder_inputs)

            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell, embedding_input,
                                                               sequence_length=tf.count_nonzero(encoder_inputs,
                                                                                                axis=1,
                                                                                                dtype=tf.int32),
                                                               dtype=tf.float32)  # time_major=True
            return encoder_outputs, encoder_state

    def __create_decoder(self, embedding_matrix, encoder_state, placeholders):
        with tf.variable_scope('decoder'):
            decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.__config.num_cells)

            embedding_input = tf.nn.embedding_lookup(embedding_matrix, placeholders.decoder_inputs)

            helper = tf.contrib.seq2seq.TrainingHelper(embedding_input,
                                                       tf.count_nonzero(placeholders.decoder_inputs,
                                                                        axis=1,
                                                                        dtype=tf.int32))  # time_major=True

            projection_layer = layers_core.Dense(len(self.__vocabulary), use_bias=False)

            decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell,
                                                      helper,
                                                      encoder_state,
                                                      output_layer=projection_layer)

            # output_time_major=True
            final_outputs, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder)

            logits = final_outputs.rnn_output

            return logits

    def __create_embedding(self):
        with tf.variable_scope('embedding'):
            embedding_matrix = tf.get_variable('embedding_matrix',
                                               [len(self.__vocabulary), self.__config.embedding_size],
                                               dtype=tf.float32)
            return embedding_matrix

    def create_model(self):

        placeholders = self.__create_placeholders()

        embedding_matrix = self.__create_embedding()

        encoder_outputs, encoder_state = self.__create_encoder(embedding_matrix, placeholders.encoder_inputs)

        logits = self.__create_decoder(embedding_matrix, encoder_state, placeholders)

        Model = namedtuple('Model', ['placeholders', 'logits'])

        return Model(
            placeholders=placeholders,
            logits=logits
        )