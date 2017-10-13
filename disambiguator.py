import tensorflow as tf
import numpy as np
import pandas as pd

from data_processing.analyses_processor import AnalysesProcessor
from data_processing.train_data_processor import TrainDataProcessor
from model.model_inference import BuildInferenceModel
from model.model_train import BuildTrainModel


class Disambiguator(object):
    def __init__(self, model_configuration):
        self.__config = model_configuration

        # Loading train data metadata
        self.__train_data_processor = TrainDataProcessor(model_configuration)

        self.__analyses_processor = AnalysesProcessor(model_configuration, self.__train_data_processor.vocabulary)

        train_dataset_metadata = self.__train_data_processor.load_dataset_metadata()

        # Building train model
        build_train_model = BuildTrainModel(model_configuration,
                                            self.__train_data_processor.vocabulary,
                                            self.__train_data_processor.inverse_vocabulary,
                                            train_dataset_metadata)

        build_train_model.create_model()

        # Building inference model
        build_inference_model = BuildInferenceModel(model_configuration,
                                                    self.__train_data_processor.inverse_vocabulary,
                                                    train_dataset_metadata,
                                                    build_train_model)

        self.__inference_model = build_inference_model.create_model()

    def __logits_to_sequence_probability(self, logits):
        sum_max_logits = np.sum(logits.max(axis=1))
        exp_sum_max_logits = np.exp(sum_max_logits)
        return exp_sum_max_logits / (exp_sum_max_logits+1)

    def create_analysis_window_batch_generator(self):
        self.__analyses_processor.build_analyses_dataframe_from_file(self.__config.analyses_path)

        corpus_dataframe = self.__train_data_processor.corpus_dataframe
        continue_count = 0
        for index, row in  corpus_dataframe.iterrows():
            # Stop before the end
            if index + self.__config.window_length >= corpus_dataframe.shape[0]: break

            # We still have a NaN in the current window
            if continue_count > 0:
                continue_count-=1
                continue

            # The end of the current window is a NaN
            if pd.isnull(corpus_dataframe.loc[index + self.__config.window_length]['word']):
                continue_count = self.__config.window_length
                continue

            print('Window starting from ', index)
            # Collecting analyses for each word in the current window (list of lists)
            list_of_combinational_analyses = []
            for id_in_window in range(index, index + self.__config.window_length):
                word = corpus_dataframe.loc[id_in_window]['word']
                analyses_of_word = self.__analyses_processor.get_analyses_list_for_word(word)
                list_of_combinational_analyses.append(analyses_of_word)
                print('\t', word, '\t', analyses_of_word)


    def analysis_window_batches_to_probabilities(self, analysis_window_batch_generator):
        with self.__inference_model.graph.as_default():
            with tf.Session() as sess:
                saver = tf.train.Saver()

                sess.run(tf.tables_initializer())
                sess.run(tf.global_variables_initializer())

                saver.restore(sess, self.__config.train_files_save_model)

                for analysis_window_batch in analysis_window_batch_generator:
                    final_outputs = sess.run([self.__inference_model.final_outputs],
                                             feed_dict={
                                                 self.__inference_model.placeholders.infer_inputs: analysis_window_batch
                                             })

                    yield map(self.__logits_to_sequence_probability, final_outputs[0].rnn_output)
