import glob
import os
from collections import namedtuple
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.python.layers import core as layers_core
import re


class MorphDisamTrainer(object):
    def __init__(self, train_files_paths, config):
        self.__config = config

        self.__vocabulary = self.__read_features_to_vocabulary(train_files_paths.tags, train_files_paths.roots)
        self.__corpus_dataframe = self.__read_corpus_dataframe(train_files_paths.corpus)

    def __read_features_to_vocabulary(self, file_tags, file_roots):
        features = []
        with open(file_tags, encoding='utf-8') as f: features.extend(f.read().splitlines())
        with open(file_roots, encoding='utf-8') as f: features.extend(f.read().splitlines())
        return dict(zip(features, range(self.__config.vocabulary_start_index,
                                        len(features) + self.__config.vocabulary_start_index)))

    def __read_corpus_dataframe(self, path_corpuses):
        return pd.concat(
            (pd.read_csv(f, sep='\t', usecols=[4], skip_blank_lines=True, header=None, nrows=self.__config.nrows,
                         names=['analysis'])
             for f in glob.glob(path_corpuses)), ignore_index=True) \
            .applymap(lambda analysis: self.__lookup_analysis_to_list(analysis))

    def __lookup_analysis_to_list(self, analysis):
        root = re.search(r'\w+', analysis, re.UNICODE).group(0)
        tags = re.findall(r'\[[^]]*\]', analysis, re.UNICODE)
        return [self.__vocabulary.get(root, self.__config.marker_unknown)] + [
            self.__vocabulary.get(tag, self.__config.marker_unknown) for tag in tags]

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

                    print('Epoch\t',epoch_id,'\t',losses)

    def generate_batches(self):

        source_sequences = []
        target_sequences = []

        self.__max_source_sequence_length = 0
        self.__max_target_sequence_length = 0

        # Gathering examples in list format
        for analysis_id in range(self.__corpus_dataframe.shape[0] - self.__config.window_length):
            source_sequence = []

            source_window_dataframe = self.__corpus_dataframe.loc[
                                      analysis_id:analysis_id + self.__config.window_length - 1]

            for source_row in source_window_dataframe['analysis']:
                source_sequence.extend(source_row + [self.__config.marker_analysis_divider])

            target_sequence = self.__corpus_dataframe.loc[analysis_id + self.__config.window_length]['analysis']

            source_sequence.extend([target_sequence[0], self.__config.marker_end_of_sentence])

            if len(source_sequence) > self.__max_source_sequence_length:
                self.__max_source_sequence_length = len(source_sequence)

            if len(target_sequence) > self.__max_target_sequence_length:
                self.__max_target_sequence_length = len(target_sequence)

            # Lists constructed
            source_sequences.append(source_sequence)
            target_sequences.append(target_sequence)

        # Padding lists
        for i, source_sequence in enumerate(source_sequences):
            source_sequences[i] = np.lib.pad(source_sequence,
                                             (0, self.__max_source_sequence_length - len(source_sequence)),
                                             'constant', constant_values=self.__config.marker_padding)

        for i, target_sequence in enumerate(target_sequences):
            target_sequences[i] = np.lib.pad(target_sequence,
                                             (0, self.__max_target_sequence_length - len(target_sequence)),
                                             'constant', constant_values=self.__config.marker_padding)

        source_input_matrix = np.matrix(source_sequences, dtype=np.int32)
        target_output_matrix = np.matrix(target_sequences, dtype=np.int32)
        target_input_matrix = np.roll(target_output_matrix, 1, axis=1)
        target_input_matrix[:, 0] = np.full((target_input_matrix.shape[0], 1), self.__config.marker_start_of_sentence, dtype=np.int32)

        # Chunking into batches
        source_input_batches = [source_input_matrix[i:i + self.__config.batch_size] for i in
                                range(source_input_matrix.shape[0] // self.__config.batch_size)]

        target_input_batches = [target_input_matrix[i:i + self.__config.batch_size] for i in
                                range(target_input_matrix.shape[0] // self.__config.batch_size)]

        target_output_batches = [target_output_matrix[i:i + self.__config.batch_size] for i in
                                range(target_output_matrix.shape[0] // self.__config.batch_size)]

        Dataset = namedtuple('Dataset', ['source_input_batches', 'target_input_batches', 'target_output_batches'])

        return Dataset(
            source_input_batches=source_input_batches,
            target_input_batches=target_input_batches,
            target_output_batches=target_output_batches
        )



def main():
    TrainFilesPaths = namedtuple('TrainFilesPaths', ['tags', 'roots', 'corpus'])

    Config = namedtuple('Config', ['embedding_size',
                                   'source_sequence_length',
                                   'target_sequence_length',
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
                                   'train_epochs'
                                   ])

    morph_disam_trainer = MorphDisamTrainer(
        TrainFilesPaths(
            tags=os.path.join('data', 'tags.txt'),
            roots=os.path.join('data', 'roots.txt'),
            corpus=os.path.join('data', 'szeged', '*')
        ),
        Config(
            embedding_size=10,
            source_sequence_length=50,
            target_sequence_length=10,
            num_cells=32,
            batch_size=6,
            window_length=5,
            marker_padding=0,
            marker_analysis_divider=1,
            marker_start_of_sentence=2,
            marker_end_of_sentence=3,
            marker_unknown=4,
            vocabulary_start_index=5,
            nrows=5,
            max_gradient_norm=1,  # 1..5
            learning_rate=0.001,
            train_epochs=100
        )
    )

    dataset = morph_disam_trainer.generate_batches()

    placeholders, logits = morph_disam_trainer.create_network()

    morph_disam_trainer.train(placeholders, logits, dataset)


if __name__ == '__main__':
    main()
