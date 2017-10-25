import numpy as np
import tensorflow as tf
import math

def input_to_output(inference_model, input, config, vocabulary, inverse_vocabulary):

    input_words = input.split(' ')

    input_vector = [vocabulary.get(word, config.marker_unknown) for word in input_words]
    input_vector += [config.marker_padding]*(config.max_source_sequence_length-len(input_vector))

    with inference_model.graph.as_default():
        with tf.Session() as sess:
            saver = tf.train.Saver()

            sess.run(tf.tables_initializer())
            sess.run(tf.global_variables_initializer())

            saver.restore(sess, config.train_files_save_model)

            final_outputs = sess.run(
                [inference_model.final_outputs],
                feed_dict={
                    inference_model.placeholders.infer_inputs: np.matrix(input_vector)
                }
            )

            print(" ".join([inverse_vocabulary.get(word_id, inverse_vocabulary[config.marker_unknown]) for word_id in final_outputs[0].sample_id[0]]))
            print(final_outputs[0].rnn_output[0].shape)
            print(np.max(final_outputs[0].rnn_output[0], axis=1))
            print(math.e ** np.sum(np.log(np.max(final_outputs[0].rnn_output[0], axis=1))))
