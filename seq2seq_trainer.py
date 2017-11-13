import os
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from utils import Utils
import numpy as np

class Seq2SeqTrainer(object):
    def __init__(self, train_graph, config, model):
        self.__config = config
        self.__model = model
        self.__train_graph = train_graph
        self.__train_session = tf.Session(graph=self.__train_graph)

    def train(self, data_processor, train_dataframes):
        print('def train(self, dataset):')

        with self.__train_graph.as_default():
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

                train_loss = tf.reduce_sum(cross_entropy * target_weights) / tf.to_float(self.__config.batch_size)

                # Calculate and clip gradients
                params = tf.trainable_variables()
                gradients = tf.gradients(train_loss, params)
                clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.__config.max_gradient_norm)

                # Trainable parameters count
                total_parameters = 0
                for variable in params:
                    # shape is an array of tf.Dimension
                    shape = variable.get_shape()
                    variable_parameters = 1
                    for dim in shape:
                        variable_parameters *= dim.value
                    total_parameters += variable_parameters
                print('\t#{trainable parameters}:',total_parameters)

                # Mistakes
                correct_elements = tf.equal(labels, self.__model.output_sequences * tf.cast(target_weights,dtype=tf.int32))
                batch_accuracy = tf.reduce_mean(tf.cast(correct_elements, tf.float32))

                # Optimization
                global_step = tf.Variable(0, trainable=False)
                if self.__config.train_decaying_learning_rate:
                    if self.__config.train_decay_type == 'piecewise_constant':
                        boundaries = [10000, 50000]
                        learning_rates = [0.001, 0.001]
                        learning_rate = tf.train.piecewise_constant(global_step, boundaries, learning_rates)
                    else:
                        learning_rate = tf.train.exponential_decay(
                            self.__config.train_starter_learning_rate,
                            global_step,
                            self.__config.train_decay_steps,
                            self.__config.train_decay_rate,
                            staircase=False
                        )
                else:
                    learning_rate = tf.constant(self.__config.train_starter_learning_rate)

                optimizer = getattr(tf.train, self.__config.train_loss_optimizer)(learning_rate, **self.__config.train_loss_optimizer_kwargs)
                update_step = optimizer.apply_gradients(zip(clipped_gradients, params), global_step=global_step)

            # Getting saver
            saver = tf.train.Saver()

            # Logging
            tf.summary.scalar('loss', train_loss)
            tf.summary.scalar('learning_rate', learning_rate)
            tf.summary.scalar('global_step', global_step)
            tf.summary.scalar('train_batch_accuracy', batch_accuracy)
            projector_config = projector.ProjectorConfig()
            embedding = projector_config.embeddings.add()
            embedding.tensor_name = self.__model.embedding_matrix.name
            embedding.metadata_path = self.__config.embedding_labels_metadata
            merged_summary_op = tf.summary.merge_all()

            summary_writer = tf.summary.FileWriter(self.__config.model_directory, graph=tf.get_default_graph())

            # Restore network if necessary
            if self.__config.train_continue_previous:
                saver.restore(self.__train_session, os.path.join(self.__config.model_directory, self.__config.model_name))
            else:
                #self.__train_session.run(tf.tables_initializer())
                self.__train_session.run(tf.global_variables_initializer())


            stopping_step = 0
            lowest_loss = None
            should_stop = False
            try:
                for epoch_id in range(1, self.__config.train_epochs+1):
                    source_input_examples, target_input_examples, target_output_examples = data_processor.get_train_examples_matrices(train_dataframes)

                    #total_batches = int(source_input_examples.shape[0] / self.__config.batch_size)
                    total_batches = source_input_examples.shape[0] // self.__config.batch_size
                    print('\t#{total batches}:',total_batches)

                    print('\tRunning epoch ', str(epoch_id)+'/'+str(self.__config.train_epochs))
                    losses = []

                    batch_accuracies = []
                    batch_ptr = 0
                    for batch_id in range(total_batches):
                        source_input_batch = source_input_examples[batch_ptr:batch_ptr + self.__config.batch_size]
                        target_input_batch = target_input_examples[batch_ptr:batch_ptr+self.__config.batch_size]
                        target_output_batch = target_output_examples[batch_ptr:batch_ptr+self.__config.batch_size]

                        if self.__config.train_shuffle_examples_in_batches:
                            randomize = np.arange(self.__config.batch_size)
                            source_input_batch = source_input_batch[randomize]
                            target_input_batch = target_input_batch[randomize]
                            target_output_batch = target_output_batch[randomize]

                        summary_return, \
                        update_step_return, \
                        train_loss_return, \
                        batch_accuracy_return, \
                        decayed_learning_rate_return,\
                        global_step_return = self.__train_session.run(
                            [merged_summary_op,
                             update_step,
                             train_loss,
                             batch_accuracy,
                             learning_rate,
                             global_step],
                            feed_dict={
                                self.__model.placeholders.encoder_inputs: source_input_batch,
                                self.__model.placeholders.decoder_inputs: target_input_batch,
                                self.__model.placeholders.decoder_outputs: target_output_batch
                            })

                        batch_accuracies.append(batch_accuracy_return * 100)
                        accuracy_so_far = sum(batch_accuracies) / len(batch_accuracies)
                        losses.append(train_loss_return)
                        average_train_loss_for_batch = sum(losses) / len(losses)
                        Utils.update_progress(batch_id+1, total_batches, "Acc:%-5.2f LR:%-8f GS:%-7d AVGLOSS:%g" % (accuracy_so_far, decayed_learning_rate_return, global_step_return, average_train_loss_for_batch))
                        #Utils.update_progress(batch_id+1, total_batches, 'ABO: ' + str(round(, 5)) + '%, Learning rate: '+str(decayed_learning_rate_return)+ ', Global step: '+str(global_step_return))
                        summary_writer.add_summary(summary_return, global_step_return)
                        batch_ptr += self.__config.batch_size

                    avg_loss = sum(losses)/len(losses)

                    if lowest_loss is None:
                        lowest_loss = avg_loss

                    print('\n\tLosses - min: %-12g max: %-12g avg: %-12g\n' % (min(losses), max(losses), avg_loss))

                    if avg_loss < lowest_loss:
                        stopping_step = 0
                        lowest_loss = avg_loss
                    else:
                        stopping_step+=1

                    if self.__config.train_early_stop_after_not_decreasing_loss_num is not None and stopping_step >= self.__config.train_early_stop_after_not_decreasing_loss_num:
                        should_stop = True
                        print('\tEarly stopping is triggered!')

                    if (epoch_id > 0 and epoch_id % self.__config.train_save_modulo == 0) or should_stop:
                        print('\tSaving model... ', end='')
                        save_path = saver.save(self.__train_session, os.path.join(self.__config.model_directory, self.__config.model_name))
                        projector.visualize_embeddings(summary_writer, projector_config)
                        print('\tSAVED to: ', save_path)
                        if should_stop:
                            break
            except KeyboardInterrupt:
                print('\n\nTraining interrupted by user!')

            print('\nLowest loss:', lowest_loss)