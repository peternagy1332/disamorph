import tensorflow as tf


class BuildInferenceModel(object):
    def __init__(self, model_configuration, inverse_vocabulary):
        self.__config = model_configuration
        self.__inverse_vocabulary = inverse_vocabulary

    def __lookup_vector_to_analysis(self, vector):
        analysis = ''

        list = vector.tolist() # TODO: flatten() did not work

        for component in list[0]:
            analysis += self.__inverse_vocabulary[component]

        return analysis

    def infer(self, input_sequence):
        return self.__lookup_vector_to_analysis(input_sequence)
        # saver = tf.train.Saver()
        #
        # with tf.Session as sess:
        #
        #
        #     saver.restore(sess, self.__config.train_files_save_model)
