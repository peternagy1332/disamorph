import tensorflow as tf

from utils import Utils


class MorphDisamTrainer(object):
    def __init__(self, train_graph, config, model):
        self.__config = config
        self.__model = model
        self.__train_graph = train_graph

    def train(self, source_input_examples, target_input_examples, target_output_examples):
        print('def train(self, dataset):')

        with self.__train_graph.as_default():
            with tf.variable_scope('train'):
                real_decoder_output_lengths = tf.count_nonzero(self.__model.placeholders.decoder_outputs, axis=1, dtype=tf.int32)
                real_max_width = tf.reduce_max(real_decoder_output_lengths)
                labels = self.__model.placeholders.decoder_outputs[:, :real_max_width]

                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=labels,
                    logits=self.__model.logits)

                target_weights = tf.sequence_mask(
                    lengths=real_decoder_output_lengths,
                    maxlen=real_max_width,
                    dtype=self.__model.logits.dtype
                )

                train_loss = tf.reduce_sum(cross_entropy * target_weights) / tf.to_float(self.__config.batch_size)

                tf.summary.scalar('loss', train_loss)

                merged_summary_op = tf.summary.merge_all()

                # Calculate and clip gradients
                params = tf.trainable_variables()
                gradients = tf.gradients(train_loss, params)
                clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.__config.max_gradient_norm)

                # Optimization
                optimizer = getattr(tf.train, self.__config.train_loss_optimizer)(self.__config.train_learning_rate)
                update_step = optimizer.apply_gradients(zip(clipped_gradients, params))

                # Mistakes
                mistakes = tf.not_equal(labels, self.__model.output_sequences)
                error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

                saver = tf.train.Saver()

                with tf.Session() as sess:
                    sess.run(tf.tables_initializer()) # TODO: What is this?
                    sess.run(tf.global_variables_initializer())

                    summary_writer = tf.summary.FileWriter(self.__config.train_files_losses, graph=tf.get_default_graph())

                    total_batches = int(source_input_examples.shape[0]/self.__config.batch_size)

                    print('\t#{batches}:', total_batches)

                    stopping_step = 0
                    lowest_loss = None
                    should_stop = False
                    for epoch_id in range(1, self.__config.train_epochs+1):
                        print('\tRunning epoch ', str(epoch_id)+'/'+str(self.__config.train_epochs))
                        losses = []

                        batch_learn_ratios = []
                        batch_ptr = 0
                        for batch_id in range(total_batches):
                            Utils.update_progress(batch_id, total_batches, 'Training on batches')
                            summary_return, update_step_return, train_loss_return, error_return = sess.run(
                                [merged_summary_op, update_step, train_loss, error],
                                feed_dict={
                                    self.__model.placeholders.encoder_inputs: source_input_examples[batch_ptr:batch_ptr + self.__config.batch_size],
                                    self.__model.placeholders.decoder_inputs: target_input_examples[batch_ptr:batch_ptr+self.__config.batch_size],
                                    self.__model.placeholders.decoder_outputs: target_output_examples[batch_ptr:batch_ptr+self.__config.batch_size]
                                })

                            batch_learn_ratios.append((1 - error_return) * 100)
                            Utils.update_progress(batch_id+1, total_batches, 'Average batch overfitting: ' + str(round(sum(batch_learn_ratios) / len(batch_learn_ratios), 5)) + '%')
                            summary_writer.add_summary(summary_return, epoch_id * total_batches + batch_id)
                            losses.append(train_loss_return)
                            batch_ptr += self.__config.batch_size


                        avg_loss = sum(losses)/len(losses)

                        if lowest_loss is None:
                            lowest_loss = avg_loss

                        print('\n\tLosses -  min: ' + str(round(min(losses),5))+ ', max: '+ str(round(max(losses),4))+ ', avg: ' + str(round(avg_loss,5)), '\n')

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
                            save_path = saver.save(sess, self.__config.train_files_save_model)
                            print('\tSAVED to: ', save_path)
                            if should_stop:
                                break

                    print(lowest_loss)

    def evaluate_model(self, disambiguator, sentence_dataframes, every_nth_sentence = None):
        accuracies = []
        print('\t#{sentences}: ', len(sentence_dataframes[::every_nth_sentence]))
        print('\tSenten.\t#{words}\tAccuracy')
        for test_sentence_id, test_sentence_dataframe in enumerate(sentence_dataframes[::every_nth_sentence]):
            words_to_disambiguate = test_sentence_dataframe['word'].tolist()  # Including <SOS>

            disambiguated_sentence = next(disambiguator.disambiguate_words_by_sentence_generator(words_to_disambiguate))
            correct_analyses = test_sentence_dataframe.loc[self.__config.window_length - 1:]['correct_analysis'].tolist()

            matching_analyses = 0
            for i in range(len(disambiguated_sentence)):
                if disambiguated_sentence[i] == correct_analyses[i]:
                    matching_analyses += 1

            accuracy = 100 * matching_analyses / len(words_to_disambiguate)
            accuracies.append(accuracy)
            print('\t'+str(test_sentence_id)+ '\t'+ str(len(correct_analyses))+ '\t\t'+ str(round(accuracy,5))+'%')

        print('\tAccuracies - min: '+ str(round(min(accuracies),5))+ '%\tmax: '+ str(round(max(accuracies),5))+ '%\tavg: '+ str(round(sum(accuracies)/len(accuracies),5))+'%')