from collections import namedtuple

from tensorflow.contrib.tensorboard.plugins import projector
import tensorflow as tf

from model.model_train import BuildTrainModel


class BuildValidationModel(object):
    def __init__(self, model_configuration, analyses_processor):
        self.__config = model_configuration

        self.__build_train_model = BuildTrainModel(model_configuration, analyses_processor)

        self.__graph = tf.Graph()

    def create_model(self):
        self.__train_model = self.__build_train_model.create_model(self.__graph)

        projector_config, merged_summary, loss, accuracy, a_global_step, loss_sum, accuracy_sum, run_options, run_metadata = self.__create_validation_summary_variables()

        Model = namedtuple('Model', [
            'placeholders',
            'projector_config',
            'merged_summary',
            'loss',
            'accuracy',
            'graph',
            'a_global_step',
            'loss_sum',
            'accuracy_sum',
            'run_options',
            'run_metadata'
        ])

        return Model(
            placeholders=self.__train_model.placeholders,
            projector_config=projector_config,
            merged_summary=merged_summary,
            loss=loss,
            accuracy=accuracy,
            graph=self.__graph,
            a_global_step=a_global_step,
            loss_sum=loss_sum,
            accuracy_sum=accuracy_sum,
            run_options=run_options,
            run_metadata=run_metadata
        )

    def __create_validation_summary_variables(self):

        with self.__graph.as_default():

            with tf.variable_scope('metrics'):
                real_decoder_output_lengths = tf.count_nonzero(self.__train_model.placeholders.decoder_outputs, axis=1, dtype=tf.int32)
                real_max_width = tf.reduce_max(real_decoder_output_lengths)
                labels = self.__train_model.placeholders.decoder_outputs[:, :real_max_width]

                # Loss
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

                # Accuracy
                correct_elements = tf.equal(labels, self.__train_model.output_sequences * tf.cast(target_weights, dtype=tf.int32))
                #accuracy = tf.multiply(tf.reduce_mean(tf.cast(correct_elements, tf.float32)), tf.constant(100.0, dtype=tf.float32))
                accuracy = tf.multiply(tf.reduce_mean(tf.cast(correct_elements, tf.float32)), tf.constant(100.0, dtype=tf.float32))

                # Averaging
                loss_sum = tf.placeholder(tf.float32, shape=(), name='loss_sum')
                accuracy_sum = tf.placeholder(tf.float32, shape=(), name='accuracy_sum')
                a_global_step = tf.placeholder(tf.float32, shape=(), name='a_global_step')

                loss_sum2 = tf.add(loss_sum, loss)
                accuracy_sum2 = tf.add(accuracy_sum, accuracy)

                avg_loss = tf.divide(loss_sum2, a_global_step)
                avg_accuracy = tf.divide(accuracy_sum2, a_global_step)

                # Logging
                tf.summary.scalar('loss', loss)
                tf.summary.scalar('accuracy', accuracy)
                tf.summary.scalar('avg_loss', avg_loss)
                tf.summary.scalar('avg_accuracy', avg_accuracy)
                tf.summary.scalar('global_step', a_global_step)

                projector_config = projector.ProjectorConfig()
                embedding = projector_config.embeddings.add()
                embedding.tensor_name = self.__train_model.embedding_matrix.name
                embedding.metadata_path = self.__config.data_label_metadata_file

                # Merge all summary operations into a single one
                merged_summary = tf.summary.merge_all()

                #global_step_inc_op = tf.add(a_global_step, tf.add(a_global_step,tf.constant(1.0, dtype=tf.float32)))

                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

                return projector_config, merged_summary, loss, accuracy, a_global_step, loss_sum, accuracy_sum, run_options, run_metadata