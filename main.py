import os
from collections import namedtuple
import tensorflow as tf
from tensorflow.python.layers import core as layers_core
from train_data_processor import TrainDataProcessor
from config import ModelConfiguration, Dataset


class MorphDisamTrainer(object):
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

            decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, encoder_state, output_layer=projection_layer)

            final_outputs, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder)  # output_time_major=True

            logits = final_outputs.rnn_output

            return logits

    def __create_embedding(self):
        with tf.variable_scope('embedding'):
            embedding_matrix = tf.get_variable('embedding_matrix',
                                               [len(self.__vocabulary), self.__config.embedding_size],
                                               dtype=tf.float32)
            return embedding_matrix

    def create_network(self):

        placeholders = self.__create_placeholders()

        embedding_matrix = self.__create_embedding()

        encoder_outputs, encoder_state = self.__create_encoder(embedding_matrix, placeholders.encoder_inputs)

        logits = self.__create_decoder(embedding_matrix, encoder_state, placeholders)

        return placeholders, logits

    def train(self, placeholders, logits, dataset):
        with tf.variable_scope('train'):
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=placeholders.decoder_outputs,
                                                                           logits=logits)

            target_weights = tf.sequence_mask(tf.count_nonzero(placeholders.decoder_outputs, axis=1, dtype=tf.int32),
                                              placeholders.decoder_outputs.shape[1].value,
                                              dtype=logits.dtype)

            train_loss = tf.reduce_sum(cross_entropy * target_weights) / self.__config.batch_size

            # Calculate and clip gradients
            params = tf.trainable_variables()
            gradients = tf.gradients(train_loss, params)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.__config.max_gradient_norm)

            # Optimization
            optimizer = tf.train.GradientDescentOptimizer(self.__config.learning_rate)  # AdamOptimizer
            update_step = optimizer.apply_gradients(zip(clipped_gradients, params))

            #merged_summary = tf.summary.merge_all()

            with tf.Session() as sess:
                sess.run(tf.tables_initializer())

                sess.run(tf.global_variables_initializer())

                for epoch_id in range(self.__config.train_epochs):
                    losses = []
                    for batch_id in range(len(dataset.source_input_batches)):
                        update_step_return, train_loss_return = sess.run([update_step, train_loss], feed_dict={
                            placeholders.encoder_inputs: dataset.source_input_batches[batch_id],
                            placeholders.decoder_inputs: dataset.target_input_batches[batch_id],
                            placeholders.decoder_outputs: dataset.target_output_batches[batch_id]
                        })

                        losses.append(train_loss_return)

                    print('Epoch\t',epoch_id,'\t','Losses: ',losses)



def main():
    # Should be loaded from YAML
    model_configuration = ModelConfiguration()
    model_configuration.embedding_size=10
    model_configuration.num_cells=32
    model_configuration.batch_size=6
    model_configuration.window_length=5
    model_configuration.marker_padding=0
    model_configuration.marker_analysis_divider=1
    model_configuration.marker_start_of_sentence=2
    model_configuration.marker_end_of_sentence=3
    model_configuration.marker_unknown=4
    model_configuration.vocabulary_start_index=5
    model_configuration.nrows=5
    model_configuration.max_gradient_norm=1  # 1..5
    model_configuration.learning_rate=1
    model_configuration.train_epochs=100
    model_configuration.train_files_tags=os.path.join('data', 'tags.txt')
    model_configuration.train_files_roots=os.path.join('data', 'roots.txt')
    model_configuration.train_files_corpus=os.path.join('data', 'szeged', '*')

    # Loading train data
    train_data_processor = TrainDataProcessor(model_configuration)
    dataset, max_source_sequence_length, max_target_sequence_length, vocabulary = train_data_processor.process_dataset()

    # Building graph
    morph_disam_trainer = MorphDisamTrainer(model_configuration,
                                            vocabulary,
                                            max_source_sequence_length,
                                            max_target_sequence_length)

    placeholders, logits = morph_disam_trainer.create_network()

    # Begin training
    morph_disam_trainer.train(placeholders, logits, dataset)


if __name__ == '__main__':
    main()
