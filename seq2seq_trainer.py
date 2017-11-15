import os
import tensorflow as tf
import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector
from utils import Utils

class Seq2SeqTrainer(object):
    def __init__(self, train_graph, model_configuration, model, utils):
        self.__config = model_configuration
        self.__model = model
        self.__utils = utils
        self.__train_graph = train_graph
        self.__train_session = tf.Session(graph=self.__train_graph)

        self.__global_minimum_loss = None

    def __create_train_variables(self):
        with tf.variable_scope('train'):
            real_decoder_output_lengths = tf.count_nonzero(self.__model.placeholders.decoder_outputs, axis=1, dtype=tf.int32)
            real_max_width = tf.reduce_max(real_decoder_output_lengths)
            labels = self.__model.placeholders.decoder_outputs[:, :real_max_width]

            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels,
                logits=self.__model.logits
            )

            target_weights = tf.sequence_mask(
                lengths=real_decoder_output_lengths,
                maxlen=real_max_width,
                dtype=self.__model.logits.dtype
            )

            loss = tf.reduce_sum(cross_entropy * target_weights) / tf.to_float(self.__config.train_batch_size)

            # Calculate and clip gradients
            params = tf.trainable_variables()
            gradients = tf.gradients(loss, params)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.__config.network_max_gradient_norm)

            # Trainable parameters count
            total_parameters = 0
            for variable in params:
                # shape is an array of tf.Dimension
                shape = variable.get_shape()
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value
                total_parameters += variable_parameters
            print('\t#{trainable parameters}:', total_parameters)

            # Accuracy
            correct_elements = tf.equal(labels, self.__model.output_sequences * tf.cast(target_weights, dtype=tf.int32))
            accuracy = tf.reduce_mean(tf.cast(correct_elements, tf.float32))

            # Optimization
            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.piecewise_constant(
                global_step,
                list(map(lambda d: d['until_global_step'], self.__config.train_schedule)),
                list(map(lambda d: d['learning_rate'], self.__config.train_schedule))
            )

            optimizer = getattr(tf.train, self.__config.train_loss_optimizer)(learning_rate, **self.__config.train_loss_optimizer_kwargs)
            update_step = optimizer.apply_gradients(zip(clipped_gradients, params), global_step=global_step)

            # Logging
            tf.summary.scalar('loss', loss)
            tf.summary.scalar('learning_rate', learning_rate)
            tf.summary.scalar('global_step', global_step)
            tf.summary.scalar('accuracy', accuracy)

            # Merge all summary operations into a single one
            merged_summary = tf.summary.merge_all()

            return loss, learning_rate, global_step, accuracy, update_step, merged_summary

    def __create_validation_variables(self):
        with tf.variable_scope('validation'):
            real_decoder_output_lengths = tf.count_nonzero(self.__model.placeholders.decoder_outputs, axis=1, dtype=tf.int32)
            real_max_width = tf.reduce_max(real_decoder_output_lengths)
            labels = self.__model.placeholders.decoder_outputs[:, :real_max_width]

            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels,
                logits=self.__model.logits
            )

            target_weights = tf.sequence_mask(
                lengths=real_decoder_output_lengths,
                maxlen=real_max_width,
                dtype=self.__model.logits.dtype
            )

            loss = tf.reduce_sum(cross_entropy * target_weights) / tf.to_float(self.__config.train_batch_size)

            correct_elements = tf.equal(labels, self.__model.output_sequences * tf.cast(target_weights, dtype=tf.int32))
            accuracy = tf.reduce_mean(tf.cast(correct_elements, tf.float32))

            # Logging
            tf.summary.scalar('loss', loss)
            tf.summary.scalar('accuracy', accuracy)

            projector_config = projector.ProjectorConfig()
            embedding = projector_config.embeddings.add()
            embedding.tensor_name = self.__model.embedding_matrix.name
            embedding.metadata_path = self.__config.embedding_labels_metadata

            # Merge all summary operations into a single one
            merged_summary = tf.summary.merge_all()

            return loss, accuracy, merged_summary, projector_config

    def __validate_epoch(self, validation_dataset, summary_writer, saver):
        print('\tValidating epoch...')

        loss, accuracy, merged_summary, projector_config = self.__create_validation_variables()

        total_batches = validation_dataset.source_input_examples.shape[0] // self.__config.train_batch_size
        print('\t#{total validation batches}:', total_batches)

        losses = []
        accuracies = []
        batch_ptr = 0

        for batch_id in range(total_batches):
            source_input_batch = validation_dataset.source_input_examples[batch_ptr:batch_ptr + self.__config.train_batch_size]
            target_input_batch = validation_dataset.target_input_examples[batch_ptr:batch_ptr + self.__config.train_batch_size]
            target_output_batch = validation_dataset.target_output_examples[batch_ptr:batch_ptr + self.__config.train_batch_size]

            r_loss, r_accuracy, r_merged_summary = self.__train_session.run(
                [
                    loss,
                    accuracy,
                    merged_summary
                ],
                feed_dict={
                    self.__model.placeholders.encoder_inputs: source_input_batch,
                    self.__model.placeholders.decoder_inputs: target_input_batch,
                    self.__model.placeholders.decoder_outputs: target_output_batch
                })

            accuracies.append(r_accuracy * 100)
            losses.append(r_loss)

            avgacc = sum(accuracies) / len(accuracies)
            avgloss = sum(losses) / len(losses)

            Utils.update_progress(batch_id + 1, total_batches, "AVGACC:%-5.2f BID:%-7d AVGLOSS:%g" % (avgacc, batch_id, avgloss))
            summary_writer.add_summary(r_merged_summary, batch_id)

            batch_ptr += self.__config.train_batch_size

        avg_loss = sum(losses) / len(losses)
        avg_accuracy = sum(accuracies) / len(accuracies)

        print('\n\tLosses     - min: %-12g max: %-12g avg: %-12g' % (min(losses), max(losses), avg_loss))
        print('\tAccuracies - min: %-12g max: %-12g avg: %-12g\n' % (min(accuracies), max(accuracies), avg_accuracy))

        if self.__global_minimum_loss is None:
            self.__global_minimum_loss = avg_loss
        else:
            if avg_loss < self.__global_minimum_loss:
                self.__global_minimum_loss = avg_loss
                print('\tAVGLOSS<MIN_AVGLOSS (%g<%g) => Saving model...' % (avg_loss, self.__global_minimum_loss))
                save_path = saver.save(self.__train_session, os.path.join(self.__config.model_directory, self.__config.model_name))
                projector.visualize_embeddings(summary_writer, projector_config)
                print('\tSAVED to: ', save_path, '\n')

        print()

    def __run_epochs(self, data_processor, sentence_dicts, summary_writer):

        loss, learning_rate, global_step, accuracy, update_step, merged_summary = self.__create_train_variables()

        saver = tf.train.Saver()

        # Restore network if necessary
        if self.__config.train_continue_previous:
            saver.restore(self.__train_session, os.path.join(self.__config.model_directory, self.__config.model_name))
        else:
            self.__train_session.run(tf.global_variables_initializer())

        for epoch_id in range(1, self.__config.train_epochs + 1):
            print()
            train_dataset, validation_dataset, _ = data_processor.get_example_matrices(sentence_dicts)
            self.__utils.print_elapsed_time()

            # total_batches = int(source_input_examples.shape[0] / self.__config.train_train_batch_size)
            total_batches = train_dataset.source_input_examples.shape[0] // self.__config.train_batch_size

            print('\tRunning epoch ', str(epoch_id) + '/' + str(self.__config.train_epochs))
            print('\t#{total train batches}:', total_batches)

            losses = []
            accuracies = []
            batch_ptr = 0

            for batch_id in range(total_batches):
                source_input_batch = train_dataset.source_input_examples[batch_ptr:batch_ptr + self.__config.train_batch_size]
                target_input_batch = train_dataset.target_input_examples[batch_ptr:batch_ptr + self.__config.train_batch_size]
                target_output_batch = train_dataset.target_output_examples[batch_ptr:batch_ptr + self.__config.train_batch_size]

                if self.__config.train_shuffle_examples_in_batches:
                    randomize = np.arange(self.__config.train_batch_size)
                    source_input_batch = source_input_batch[randomize]
                    target_input_batch = target_input_batch[randomize]
                    target_output_batch = target_output_batch[randomize]

                r_loss, r_learning_rate, r_global_step, r_accuracy, r_update_step, r_merged_summary = self.__train_session.run(
                [
                    loss,
                    learning_rate,
                    global_step,
                    accuracy,
                    update_step,
                    merged_summary
                 ],
                feed_dict={
                    self.__model.placeholders.encoder_inputs: source_input_batch,
                    self.__model.placeholders.decoder_inputs: target_input_batch,
                    self.__model.placeholders.decoder_outputs: target_output_batch
                })

                accuracies.append(r_accuracy * 100)
                losses.append(r_loss)

                avgacc = sum(accuracies) / len(accuracies)
                avgloss = sum(losses) / len(losses)

                Utils.update_progress(batch_id + 1, total_batches,"AVGACC:%-5.2f LR:%-8f GS:%-7d AVGLOSS:%g" % (avgacc, r_learning_rate, r_global_step, avgloss))
                summary_writer.add_summary(r_merged_summary, r_global_step)

                batch_ptr += self.__config.train_batch_size

            avg_loss = sum(losses) / len(losses)
            avg_accuracy = sum(accuracies) / len(accuracies)

            print('\n\tLosses     - min: %-12g max: %-12g avg: %-12g' % (min(losses), max(losses), avg_loss))
            print('\tAccuracies - min: %-12g max: %-12g avg: %-12g\n' % (min(accuracies), max(accuracies), avg_accuracy))

            self.__utils.print_elapsed_time()

            self.__validate_epoch(validation_dataset, summary_writer, saver)

            self.__utils.print_elapsed_time()

            print('\n','-'*50)

    def train(self, data_processor, sentence_dicts):
        print('def train(self, dataset):')

        with self.__train_graph.as_default():

            summary_writer = tf.summary.FileWriter(self.__config.model_directory, graph=tf.get_default_graph())

            try:
                self.__run_epochs(data_processor, sentence_dicts, summary_writer)

            except KeyboardInterrupt:
                print('\n\nTraining interrupted by user!')
