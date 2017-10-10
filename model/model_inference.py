from collections import namedtuple
from tensorflow.python.layers import core as layers_core
import tensorflow as tf
import numpy as np

class BuildInferenceModel(object):
    def __init__(self, model_configuration, inverse_vocabulary, dataset_metadata, build_train_model):
        self.__config = model_configuration
        self.__inverse_vocabulary = inverse_vocabulary
        self.__dataset_metadata = dataset_metadata
        self.__build_train_model = build_train_model

        self.__infer_graph = tf.Graph()

    def create_model(self):
        with self.__infer_graph.as_default():
            placeholders = self.__create_placeholders()

            embedding_matrix = self.__build_train_model.create_embedding()

            encoder_outputs, encoder_state = self.__build_train_model.create_encoder(embedding_matrix,
                                                                               placeholders.infer_inputs)

            final_outputs = self.__create_decoder(embedding_matrix, encoder_state)

            Model = namedtuple('Model', ['placeholders', 'final_outputs'])

            return Model(
                placeholders=placeholders,
                final_outputs=final_outputs
            )

    def __create_placeholders(self):
        Placeholders = namedtuple('Placeholders', ['infer_inputs'])

        with tf.variable_scope('placeholders'):
            infer_inputs = tf.placeholder(tf.int32,
                                          [self.__config.infer_size,
                                           self.__dataset_metadata.max_source_sequence_length],
                                          'infer_inputs')

            return Placeholders(
                infer_inputs=infer_inputs
            )

    def __create_decoder(self, embedding_matrix, encoder_state):
        with tf.variable_scope('decoder'):
            decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.__config.num_cells)

            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                embedding_matrix,
                tf.fill([self.__config.infer_size], self.__config.marker_start_of_sentence),
                self.__config.marker_end_of_sentence)

            projection_layer = layers_core.Dense(len(self.__inverse_vocabulary), use_bias=False)

            decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell,
                                                      helper,
                                                      encoder_state,
                                                      output_layer=projection_layer) #time_major=True

            # output_time_major=True
            final_outputs, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
                decoder, maximum_iterations=self.__config.infer_maximum_iterations)

            return final_outputs

    def __lookup_vector_to_analysis(self, vector):
        analysis = ''

        list = vector.tolist()  # TODO: flatten() did not work

        for component in list[0]:

            analysis += self.__inverse_vocabulary[component]

        return analysis

    def __logits_to_sequence_probability(self, logits):
        sum_max_logits = np.sum(logits.max(axis=1))
        exp_sum_max_logits = np.exp(sum_max_logits)
        return exp_sum_max_logits / (exp_sum_max_logits+1)

    def infer(self, infer_model, input_sequence):
        with self.__infer_graph.as_default():
            with tf.Session() as sess:
                saver = tf.train.Saver()

                sess.run(tf.tables_initializer())
                sess.run(tf.global_variables_initializer())

                saver.restore(sess, self.__config.train_files_save_model)

                final_outputs = sess.run([infer_model.final_outputs],
                                                feed_dict={infer_model.placeholders.infer_inputs: input_sequence})


                print('input_sequence', self.__lookup_vector_to_analysis(input_sequence),'\n')

                sequence_probability = self.__logits_to_sequence_probability(final_outputs[0].rnn_output[0])

                print('output sequence probability: ', sequence_probability)


                print('sample_id')
                print(self.__lookup_vector_to_analysis(final_outputs[0].sample_id))

                return final_outputs
