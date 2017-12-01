from collections import namedtuple
from tensorflow.python.layers import core as layers_core
import tensorflow as tf

from model.model_train import BuildTrainModel


class BuildInferenceModel(object):
    def __init__(self, model_configuration, analyses_processor):
        self.__config = model_configuration
        self.__analyses_processor = analyses_processor

        self.__build_train_model = BuildTrainModel(model_configuration, self.__analyses_processor)

        self.__graph = tf.Graph()

    def create_model(self):
        with self.__graph.as_default():
            placeholders = self.create_placeholders()

            embedding_matrix = self.__build_train_model.create_embedding()

            encoder_outputs, encoder_state = self.__build_train_model.create_encoder(embedding_matrix, placeholders.infer_inputs)


            logits, output_sequences = self.create_decoder(embedding_matrix, encoder_outputs, encoder_state, placeholders)

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
                                          [self.__config.data_batch_size,
                                           self.__config.network_max_source_sequence_length],
                                          'infer_inputs')

            return Placeholders(
                infer_inputs=infer_inputs
            )

    def create_decoder(self, embedding_matrix, encoder_outputs, encoder_state, placeholders):
        with tf.variable_scope('decoder'):
            decoder_rnn = self.__build_train_model.create_rnn()

            projection_layer = layers_core.Dense(len(self.__analyses_processor.inverse_vocabulary), activation=tf.nn.softmax, use_bias=False)

            if self.__config.inference_decoder_type=='beam_search':
                tiled_encoder_outputs = tf.contrib.seq2seq.tile_batch(encoder_outputs, multiplier=self.__config.inference_beam_width)
                tiled_encoder_final_state = tf.contrib.seq2seq.tile_batch(encoder_state, multiplier=self.__config.inference_beam_width)
                tiled_sequence_length = tf.contrib.seq2seq.tile_batch(tf.count_nonzero(placeholders.infer_inputs,axis=1), multiplier=self.__config.inference_beam_width)

                mechanism = tf.contrib.seq2seq.LuongAttention(
                    num_units=self.__config.network_hidden_layer_cells,
                    memory=tiled_encoder_outputs,
                    memory_sequence_length=tiled_sequence_length
                )

                attention_cell = tf.contrib.seq2seq.AttentionWrapper(
                    decoder_rnn, mechanism, attention_layer_size=self.__config.network_hidden_layer_cells
                )

                decoder_initial_state = attention_cell.zero_state(
                    batch_size=self.__config.data_batch_size * self.__config.inference_beam_width,
                    dtype=tf.float32
                )

                decoder_initial_state = decoder_initial_state.clone(cell_state=tiled_encoder_final_state)

                # Define a beam-search decoder
                decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                    cell=attention_cell,
                    embedding=embedding_matrix,
                    start_tokens=tf.fill([self.__config.data_batch_size], self.__config.marker_go),
                    end_token=self.__config.marker_end_of_sentence,
                    initial_state=decoder_initial_state,
                    beam_width=self.__config.inference_beam_width,
                    output_layer=projection_layer,
                    length_penalty_weight=0.0
                )

                final_outputs, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
                    decoder, maximum_iterations=self.__config.network_max_target_sequence_length,
                    output_time_major=False, swap_memory=True
                )

                predicted_ids = final_outputs.predicted_ids
                beam_search_decoder_output = final_outputs.beam_search_decoder_output
                scores = beam_search_decoder_output.scores

                return scores, predicted_ids
            elif self.__config.inference_decoder_type == 'greedy':
                mechanism = tf.contrib.seq2seq.LuongAttention(
                    self.__config.network_hidden_layer_cells, encoder_outputs,
                    memory_sequence_length=[self.__config.network_max_target_sequence_length]*self.__config.data_batch_size,
                    scale=True
                )

                attention_cell = tf.contrib.seq2seq.AttentionWrapper(
                    decoder_rnn, mechanism, attention_layer_size=self.__config.network_hidden_layer_cells
                )

                decoder_initial_state = attention_cell.zero_state(
                    batch_size=self.__config.data_batch_size,
                    dtype=tf.float32
                )

                decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state)

                helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                    embedding_matrix,
                    tf.fill([self.__config.data_batch_size], self.__config.marker_go),
                    self.__config.marker_end_of_sentence
                )

                # Using a basic decoder
                decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell=attention_cell,
                    helper=helper,
                    initial_state=decoder_initial_state,
                    output_layer=projection_layer
                )

                final_outputs, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
                    decoder, maximum_iterations=self.__config.network_max_target_sequence_length,
                    output_time_major=False, swap_memory=True
                )

                logits = final_outputs.rnn_output
                output_sequences = final_outputs.sample_id

                return logits, output_sequences
