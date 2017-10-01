import tensorflow as tf


class MorphDisamInference(object):
    def __init__(self, model_configuration, model):
        self.__config = model_configuration
        self.__model = model

    def infer(self, input_sequence):
        saver = tf.train.Saver()

        with tf.Session as sess:


            saver.restore(sess, self.__config.train_files_save_model)
