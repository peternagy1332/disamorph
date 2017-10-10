import tensorflow as tf


class MorphDisamTrainer(object):
    def __init__(self, train_graph, config, model):
        self.__config = config
        self.__model = model
        self.__train_graph = train_graph

    def train(self, dataset):
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

                    total_batches = len(dataset.source_input_batches)

                    for epoch_id in range(1, self.__config.train_epochs+1):
                        losses = []
                        for batch_id in range(total_batches):
                            update_step_return, train_loss_return, summary_return = sess.run(
                                [update_step, train_loss, merged_summary_op],
                                feed_dict={
                                    self.__model.placeholders.encoder_inputs: dataset.source_input_batches[batch_id],
                                    self.__model.placeholders.decoder_inputs: dataset.target_input_batches[batch_id],
                                    self.__model.placeholders.decoder_outputs: dataset.target_output_batches[batch_id]
                                })

                            summary_writer.add_summary(summary_return, epoch_id * total_batches + batch_id)

                            losses.append(train_loss_return)

                        print('Epoch\t', epoch_id, '\t', 'Losses: ', losses)

                        if epoch_id > 0 and epoch_id % self.__config.train_save_modulo == 0:
                            print('Saving model... ', end='')
                            save_path = saver.save(sess, self.__config.train_files_save_model)
                            print('SAVED to: ', save_path)
