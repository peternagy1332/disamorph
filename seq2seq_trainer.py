import os
from collections import namedtuple
from tensorflow.contrib.tensorboard.plugins import projector
import tensorflow as tf
import numpy as np

from model.model_train import BuildTrainModel
from model.model_validation import BuildValidationModel
from utils import Utils

class Seq2SeqTrainer(object):
    def __init__(self, analyses_processor, model_configuration, utils):
        self.__analyses_processor = analyses_processor
        self.__config = model_configuration
        self.__utils = utils
        
        build_train_model = BuildTrainModel(model_configuration, analyses_processor)
        
        self.__train_model = build_train_model.create_model()
        self.__train_session = tf.Session(graph=self.__train_model.graph)

        build_validation_model = BuildValidationModel(model_configuration, analyses_processor)
        self.__validation_model = build_validation_model.create_model()
        self.__validation_session = tf.Session(graph=self.__validation_model.graph)

        self.__validation_global_step = 1.0
        self.__validation_loss_sum = 0.0
        self.__validation_accuracy_sum = 0.0

        self.__global_minimum_loss = None

    def __create_train_variables(self):
        with tf.variable_scope('metrics'):
            real_decoder_output_lengths = tf.count_nonzero(self.__train_model.placeholders.decoder_outputs, axis=1, dtype=tf.int32)
            real_max_width = tf.reduce_max(real_decoder_output_lengths)
            labels = self.__train_model.placeholders.decoder_outputs[:, :real_max_width]

            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels,
                logits=self.__train_model.logits
            )

            target_weights = tf.sequence_mask(
                lengths=real_decoder_output_lengths,
                maxlen=real_max_width,
                dtype=self.__train_model.logits.dtype
            )

            loss = tf.reduce_sum(cross_entropy * target_weights) / tf.to_float(self.__config.data_batch_size)

            # Calculate and clip gradients
            params = tf.trainable_variables()
            gradients = tf.gradients(loss, params)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.__config.network_max_gradient_norm)

            # Optimization
            global_step = tf.Variable(0, trainable=False, dtype=tf.int32)
            learning_rate = tf.train.piecewise_constant(
                global_step,
                list(map(lambda d: d['until_global_step'], self.__config.train_schedule)),
                list(map(lambda d: d['learning_rate'], self.__config.train_schedule))
            )

            optimizer = getattr(tf.train, self.__config.train_loss_optimizer)(learning_rate, **self.__config.train_loss_optimizer_kwargs)
            update_step = optimizer.apply_gradients(zip(clipped_gradients, params), global_step=global_step)

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

            output_sequences = self.__train_model.output_sequences * tf.cast(target_weights, dtype=tf.int32)

            # Accuracy
            correct_elements = tf.equal(labels, self.__train_model.output_sequences * tf.cast(target_weights, dtype=tf.int32))
            #accuracy = tf.reduce_mean(tf.cast(correct_elements, tf.float32))
            accuracy = tf.multiply(tf.reduce_mean(tf.cast(correct_elements, tf.float32)), tf.constant(100.0, dtype=tf.float32))

            # Averaging
            loss_sum = tf.Variable(0.0, trainable=False, dtype=tf.float32)
            accuracy_sum = tf.Variable(0.0, trainable=False, dtype=tf.float32)

            #loss_sum2 = tf.add(loss_sum, loss)
            #accuracy_sum2 = tf.add(accuracy_sum, accuracy)

            avg_loss = tf.divide(loss_sum, tf.cast(global_step, tf.float32))
            avg_accuracy = tf.divide(accuracy_sum ,tf.cast(global_step, tf.float32))

            tf.summary.scalar('loss', loss)
            tf.summary.scalar('accuracy', accuracy)
            tf.summary.scalar('avg_loss', avg_loss)
            tf.summary.scalar('avg_accuracy', avg_accuracy)
            tf.summary.scalar('learning_rate', learning_rate)
            tf.summary.scalar('global_step', global_step)

            tf.summary.histogram('logits',tf.nn.softmax(self.__train_model.logits))

            # Merge all summary operations into a single one
            merged_summary_op = tf.summary.merge_all()
            loss_inc_op = tf.assign_add(loss_sum, loss)
            accuracy_inc_op = tf.assign_add(accuracy_sum, accuracy)
            stat_ops = tf.group(loss_inc_op, accuracy_inc_op)

            return loss, learning_rate, global_step, accuracy, update_step, merged_summary_op, stat_ops, output_sequences

    def __validate_model(self, validation_dataset, train_saver, r_global_step, total_train_batches):
        with self.__validation_model.graph.as_default():

            validation_saver = tf.train.Saver()

            print('\tValidation: restoring latest train model...')
            #validation_saver.restore(self.__validation_session, os.path.join(self.__config.model_directory, 'train_checkpoints', self.__config.model_name))
            latest_checkpoint = tf.train.latest_checkpoint(os.path.join(self.__config.model_directory, 'train_checkpoints'))
            print('\tRESTORED_FROM:',latest_checkpoint)
            validation_saver.restore(self.__validation_session, latest_checkpoint)

            # with self.__train_model.graph.as_default():
            #     train_saver.restore(self.__train_session, os.path.join(self.__config.model_directory, 'train_checkpoints', self.__config.model_name))

            total_validation_batches = validation_dataset.source_input_examples.shape[0] // self.__config.data_batch_size
            print('\t#{total validation batches}:', total_validation_batches)

            if validation_dataset.source_input_examples.shape[0] < self.__config.data_batch_size:
                raise ValueError('Validation dataset is too small: validation_dataset.source_input_examples.shape[0]='+str(validation_dataset.source_input_examples.shape[0])+'<data_batch_size='+str(self.__config.data_batch_size))

            validation_losses = []
            validation_accuracies = []
            validation_batch_ptr = 0

            validation_summary_writer = tf.summary.FileWriter(os.path.join(self.__config.model_directory, 'validation_checkpoints'), graph=self.__validation_model.graph)

            for validation_batch_id in range(total_validation_batches):
                source_input_batch = validation_dataset.source_input_examples[validation_batch_ptr:validation_batch_ptr + self.__config.data_batch_size]
                target_input_batch = validation_dataset.target_input_examples[validation_batch_ptr:validation_batch_ptr + self.__config.data_batch_size]
                target_output_batch = validation_dataset.target_output_examples[validation_batch_ptr:validation_batch_ptr + self.__config.data_batch_size]

                r_val_loss, r_val_accuracy, r_val_merged_summary = self.__validation_session.run(
                    [
                        self.__validation_model.loss,
                        self.__validation_model.accuracy,
                        self.__validation_model.merged_summary
                    ],
                    feed_dict={
                        self.__validation_model.placeholders.encoder_inputs: source_input_batch,
                        self.__validation_model.placeholders.decoder_inputs: target_input_batch,
                        self.__validation_model.placeholders.decoder_outputs: target_output_batch,
                        self.__validation_model.a_global_step: self.__validation_global_step,
                        self.__validation_model.loss_sum: self.__validation_loss_sum,
                        self.__validation_model.accuracy_sum: self.__validation_accuracy_sum
                    })

                self.__validation_global_step += 1.0
                self.__validation_loss_sum+=r_val_loss
                self.__validation_accuracy_sum += r_val_accuracy

                validation_accuracies.append(r_val_accuracy)
                validation_losses.append(r_val_loss)

                avgacc = sum(validation_accuracies) / len(validation_accuracies)
                avgloss = sum(validation_losses) / len(validation_losses)

                Utils.update_progress(validation_batch_id + 1, total_validation_batches, "AVGACC:%-5.2f BID:%-7d AVGLOSS:%g" % (avgacc, validation_batch_id, avgloss))

                start_step = r_global_step-total_train_batches*self.__config.train_validation_modulo
                current_step = start_step + int((r_global_step-start_step)*((validation_batch_id+1)/total_validation_batches))

                if validation_batch_id == 0:
                    validation_summary_writer.add_summary(r_val_merged_summary, start_step)

                #if (validation_batch_id+1) % self.__config.train_validation_add_summary_modulo == 0:
                validation_summary_writer.add_summary(r_val_merged_summary, current_step)

                validation_batch_ptr += self.__config.data_batch_size

            avg_validation_loss = sum(validation_losses) / len(validation_losses)
            avg_validation_accuracy = sum(validation_accuracies) / len(validation_accuracies)

            print('\n\tValidation losses     - min: %-12g max: %-12g avg: %-12g' % (min(validation_losses), max(validation_losses), avg_validation_loss))
            print('\tValidation accuracies - min: %-12g max: %-12g avg: %-12g\n' % (min(validation_accuracies), max(validation_accuracies), avg_validation_accuracy))

            if self.__global_minimum_loss is None:
                self.__global_minimum_loss = avg_validation_loss

            if avg_validation_loss <= self.__global_minimum_loss:
                print('\tAVGLOSS<MIN_AVGLOSS (%g<%g) => Saving model...' % (avg_validation_loss, self.__global_minimum_loss))
                self.__global_minimum_loss = avg_validation_loss
                save_path = train_saver.save(self.__train_session, os.path.join(self.__config.model_directory, 'validation_checkpoints', self.__config.model_name), r_global_step)
                projector.visualize_embeddings(validation_summary_writer, self.__validation_model.projector_config)
                print('\tSAVED TO: ', save_path, '\n')

            print()

    def __run_epochs(self, data_processor, sentence_dicts):
        with self.__train_model.graph.as_default():
            loss, learning_rate, global_step, accuracy, update_step, merged_summary_op, stat_ops, output_sequences = self.__create_train_variables()

            train_saver = tf.train.Saver()

            # Restore network if necessary
            if self.__config.train_continue_previous:
                if self.__config.use_train_model:
                    print('\tRestoring model from latest train checkpoint...')
                    #train_saver.restore(self.__train_session, os.path.join(self.__config.model_directory, 'train_checkpoints', self.__config.model_name))
                    latest_checkpoint = tf.train.latest_checkpoint(os.path.join(self.__config.model_directory, 'train_checkpoints'))
                else:
                    print('\tRestoring model from latest validation checkpoint...')
                    #train_saver.restore(self.__train_session, os.path.join(self.__config.model_directory, 'validation_checkpoints', self.__config.model_name))
                    latest_checkpoint = tf.train.latest_checkpoint(os.path.join(self.__config.model_directory, 'validation_checkpoints'))

                print('\tRESTORED FROM:',latest_checkpoint)
                train_saver.restore(self.__train_session, latest_checkpoint)
            else:
                print('\tTraining new model...')
                self.__train_session.run(tf.global_variables_initializer())

            train_summary_writer = tf.summary.FileWriter(os.path.join(self.__config.model_directory, 'train_checkpoints'), graph=self.__train_model.graph)

            for epoch_id in range(self.__config.train_epochs):
                print()
                train_dataset, validation_dataset, _ = data_processor.get_example_matrices(sentence_dicts)
                self.__utils.print_elapsed_time()

                # total_batches = int(source_input_examples.shape[0] / self.__config.train_data_batch_size)
                total_train_batches = train_dataset.source_input_examples.shape[0] // self.__config.data_batch_size

                print('\tRunning epoch ', str(epoch_id) + '/' + str(self.__config.train_epochs))
                print('\t#{total train batches}:', total_train_batches)

                losses = []
                accuracies = []
                batch_ptr = 0

                r_global_step = 0

                # source_input_batch = np.load(os.path.join(self.__config.data_train_matrices, 'source_input', 'sentence_source_input_examples_0000000000.npy'))[:self.__config.data_batch_size]
                # target_input_batch = np.load(os.path.join(self.__config.data_train_matrices, 'target_input', 'sentence_target_input_examples_0000000000.npy'))[:self.__config.data_batch_size]
                # target_output_batch = np.load(os.path.join(self.__config.data_train_matrices, 'target_output', 'sentence_target_output_examples_0000000000.npy'))[:self.__config.data_batch_size]

                # source_input_batch = np.load('source_input.npy')
                # target_input_batch = np.load('target_input.npy')
                # target_output_batch = np.load('target_output.npy')

                for train_batch_id in range(total_train_batches):
                    source_input_batch = train_dataset.source_input_examples[batch_ptr:batch_ptr + self.__config.data_batch_size]
                    target_input_batch = train_dataset.target_input_examples[batch_ptr:batch_ptr + self.__config.data_batch_size]
                    target_output_batch = train_dataset.target_output_examples[batch_ptr:batch_ptr + self.__config.data_batch_size]

                    if self.__config.train_shuffle_examples_in_batches:
                        randomize = np.arange(self.__config.data_batch_size)
                        source_input_batch = source_input_batch[randomize]
                        target_input_batch = target_input_batch[randomize]
                        target_output_batch = target_output_batch[randomize]

                    r_loss, r_learning_rate, r_global_step, r_accuracy, r_update_step, r_merged_summary_op, r_stat_ops, r_output_sequences = self.__train_session.run(
                    [
                        loss,
                        learning_rate,
                        global_step,
                        accuracy,
                        update_step,
                        merged_summary_op,
                        stat_ops, output_sequences
                     ],
                    feed_dict={
                        self.__train_model.placeholders.encoder_inputs: source_input_batch,
                        self.__train_model.placeholders.decoder_inputs: target_input_batch,
                        self.__train_model.placeholders.decoder_outputs: target_output_batch
                    })

                    # target_sequence_lengths = np.count_nonzero(target_output_batch, axis=1)
                    #
                    # avg_accs = []
                    # for rowid, target_sequence_length in enumerate(target_sequence_lengths.tolist()):
                    #     output_sequence = r_output_sequences[rowid][:target_sequence_length]
                    #     target_output_sequence = target_output_batch[rowid][:target_sequence_length]
                    #     print('%-60s| %s' % ("".join(self.__analyses_processor.lookup_ids_to_features(target_output_sequence)).replace("<PAD>", ""),
                    #                        "".join(self.__analyses_processor.lookup_ids_to_features(output_sequence)).replace("<PAD>", "")))
                    #     avg_accs.append(np.mean(np.equal(output_sequence, target_output_sequence)))
                    #
                    # print('Accuracy:', sum(avg_accs) * 100 / len(avg_accs),'\n')

                    accuracies.append(r_accuracy)
                    losses.append(r_loss)

                    avgacc = sum(accuracies) / len(accuracies)
                    avgloss = sum(losses) / len(losses)

                    Utils.update_progress(train_batch_id + 1, total_train_batches,"AVGACC:%-5.2f LR:%-8f GS:%-7d AVGLOSS:%g" % (avgacc, r_learning_rate, r_global_step, avgloss))
                    if train_batch_id % self.__config.train_add_summary_modulo == 0:
                        train_summary_writer.add_summary(r_merged_summary_op, (r_global_step-1))

                    batch_ptr += self.__config.data_batch_size

                avg_loss = sum(losses) / len(losses)
                avg_accuracy = sum(accuracies) / len(accuracies)

                print('\n\tLosses     - min: %-12g max: %-12g avg: %-12g' % (min(losses), max(losses), avg_loss))
                print('\tAccuracies - min: %-12g max: %-12g avg: %-12g\n' % (min(accuracies), max(accuracies), avg_accuracy))

                self.__utils.print_elapsed_time()

                if (epoch_id+1) % self.__config.train_validation_modulo == 0:
                    print('\n\tSaving current train model...')
                    train_saver.save(self.__train_session, os.path.join(self.__config.model_directory, 'train_checkpoints', self.__config.model_name), r_global_step)
                    train_saver.save(self.__train_session, os.path.join(self.__config.model_directory, 'train_checkpoints'))
                    self.__validate_model(validation_dataset, train_saver, r_global_step, total_train_batches)

                    self.__utils.print_elapsed_time()

                print('\n','-'*50)

    def train(self, data_processor, sentence_dicts):
        print('def train(self, dataset):')

        try:
            self.__run_epochs(data_processor, sentence_dicts)

        except KeyboardInterrupt:
            print('\n\nTraining interrupted by user!')
