import tensorflow as tf

from utils import Utils


class MorphDisamTrainer(object):
    def __init__(self, train_graph, config, model):
        self.__config = config
        self.__model = model
        self.__train_graph = train_graph

    def train(self, train_batches):
        print('def train(self, dataset):')

        with self.__train_graph.as_default():
            with tf.variable_scope('train'):
                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.__model.placeholders.decoder_outputs,
                    logits=self.__model.logits)

                target_weights = tf.sequence_mask(
                    tf.count_nonzero(
                        self.__model.placeholders.decoder_outputs,
                        axis=1,
                        dtype=tf.int32),
                    self.__model.placeholders.decoder_outputs.shape[1].value,
                    dtype=self.__model.logits.dtype)

                train_loss = tf.reduce_sum(cross_entropy * target_weights) / self.__config.batch_size

                tf.summary.scalar('loss', train_loss)

                merged_summary_op = tf.summary.merge_all()

                # Calculate and clip gradients
                params = tf.trainable_variables()
                gradients = tf.gradients(train_loss, params)
                clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.__config.max_gradient_norm)

                # Optimization
                optimizer = tf.train.GradientDescentOptimizer(self.__config.learning_rate)  # AdamOptimizer
                update_step = optimizer.apply_gradients(zip(clipped_gradients, params))

                saver = tf.train.Saver()

                with tf.Session() as sess:
                    sess.run(tf.tables_initializer()) # TODO: What is this?
                    sess.run(tf.global_variables_initializer())

                    summary_writer = tf.summary.FileWriter(self.__config.train_files_losses, graph=tf.get_default_graph())

                    total_batches = len(train_batches.source_input_batches)

                    stopping_step = 0
                    lowest_loss = None
                    should_stop = False
                    for epoch_id in range(1, self.__config.train_epochs+1):
                        print('\tRunning epoch ', str(epoch_id)+'/'+str(self.__config.train_epochs))
                        losses = []

                        for batch_id in range(total_batches):
                            Utils.update_progress(batch_id, total_batches, 'Training on batches...')
                            summary_return, update_step_return, train_loss_return = sess.run(
                                [merged_summary_op, update_step, train_loss],
                                feed_dict={
                                    self.__model.placeholders.encoder_inputs: train_batches.source_input_batches[batch_id],
                                    self.__model.placeholders.decoder_inputs: train_batches.target_input_batches[batch_id],
                                    self.__model.placeholders.decoder_outputs: train_batches.target_output_batches[batch_id]
                                })

                            Utils.update_progress(batch_id+1, total_batches, 'Training on batches...')
                            summary_writer.add_summary(summary_return, epoch_id * total_batches + batch_id)
                            losses.append(train_loss_return)

                        avg_loss = sum(losses)/len(losses)

                        if lowest_loss is None:
                            lowest_loss = avg_loss

                        print('\n\tLosses -  min: ' + str(round(min(losses),2))+ ', max: '+ str(round(max(losses),2)), ', avg: ' + str(round(avg_loss,2)), '\n')

                        if avg_loss < lowest_loss:
                            stopping_step = 0
                            lowest_loss = avg_loss
                        else:
                            stopping_step+=1

                        if stopping_step >= self.__config.train_early_stop_after_not_decreasing_loss_num:
                            should_stop = True
                            print('\tEarly stopping is triggered!')

                        if (epoch_id > 0 and epoch_id % self.__config.train_save_modulo == 0) or should_stop:
                            print('\tSaving model... ', end='')
                            save_path = saver.save(sess, self.__config.train_files_save_model)
                            print('\tSAVED to: ', save_path)
                            if should_stop:
                                break

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
            print('\t'+str(test_sentence_id)+ '\t'+ str(len(correct_analyses))+ '\t'+ str(round(accuracy,2))+'%')

        print('\tAccuracies: min - '+ str(round(min(accuracies),2))+ '%\tmax - '+ str(round(max(accuracies),2))+ '%\t avg - '+ str(round(sum(accuracies)/len(accuracies),2))+'%')