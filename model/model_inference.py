from collections import namedtuple
from tensorflow.python.layers import core as layers_core
import tensorflow as tf


class BuildInferenceModel(object):
    def __init__(self, model_configuration, inverse_vocabulary, build_train_model):
        self.__config = model_configuration
        self.__inverse_vocabulary = inverse_vocabulary
        self.__build_train_model = build_train_model

        self.__graph = tf.Graph()

    def create_model(self):
        with self.__graph.as_default():
            placeholders = self.create_placeholders()

            embedding_matrix = self.__build_train_model.create_embedding()

            encoder_outputs, encoder_state = self.__build_train_model.create_encoder(embedding_matrix, placeholders.infer_inputs)

            logits, output_sequences = self.create_decoder(embedding_matrix, encoder_outputs, encoder_state)

            Model = namedtuple('Model', ['placeholders', 'logits', 'output_sequences', 'graph'])

            return Model(
                placeholders=placeholders,
                logits=logits,
                output_sequences=output_sequences,
                graph=self.__graph
            )

    def create_placeholders(self):
        Placeholders = namedtuple('Placeholders', ['infer_inputs'])

        with tf.variable_scope('placeholders'):
            infer_inputs = tf.placeholder(tf.int32,
                                          [self.__config.inference_batch_size,
                                           self.__config.network_max_source_sequence_length],
                                          'infer_inputs')

            return Placeholders(
                infer_inputs=infer_inputs
            )

    def create_decoder(self, embedding_matrix, encoder_outputs, encoder_state):
        with tf.variable_scope('decoder'):
            decoder_rnn = self.__build_train_model.create_rnn()

            #decoder_inputs_sequence_length = tf.count_nonzero(placeholders.decoder_inputs, axis=1, dtype=tf.int32)

            mechanism = tf.contrib.seq2seq.LuongAttention(
                self.__config.network_hidden_layer_cells, encoder_outputs,
                memory_sequence_length=[self.__config.network_max_target_sequence_length]*self.__config.inference_batch_size,
                scale=False
            )

            decoder_rnn = tf.contrib.seq2seq.AttentionWrapper(
                decoder_rnn, mechanism, attention_layer_size=self.__config.network_hidden_layer_cells
            )

            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                embedding_matrix,
                tf.fill([self.__config.inference_batch_size], self.__config.marker_go),
                self.__config.marker_end_of_sentence
            )

            projection_layer = layers_core.Dense(len(self.__inverse_vocabulary), activation=tf.nn.softmax, use_bias=False)

            decoder_initial_state = decoder_rnn.zero_state(
                self.__config.inference_batch_size, tf.float32) \
                .clone(cell_state=encoder_state)

            decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=decoder_rnn,
                helper=helper,
                initial_state=decoder_initial_state,
                output_layer=projection_layer
            )

            # decoder_initial_state = tf.contrib.seq2seq.tile_batch(
            #     encoder_state, multiplier=1)
            #
            # # Define a beam-search decoder
            # decoder = tf.contrib.seq2seq.BeamSearchDecoder(
            #     cell=decoder_rnn,
            #     embedding=embedding_matrix,
            #     start_tokens=tf.fill([self.__config.inference_batch_size], self.__config.marker_go),
            #     end_token=self.__config.marker_end_of_sentence,
            #     initial_state=decoder_initial_state,
            #     beam_width=10,
            #     output_layer=projection_layer,
            #     length_penalty_weight=0.0)

            # output_time_major=True
            final_outputs, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
                decoder, maximum_iterations=self.__config.network_max_target_sequence_length,
                output_time_major=False, swap_memory=True
            )

            logits = final_outputs.rnn_output
            output_sequences = final_outputs.sample_id

            return logits, output_sequences
