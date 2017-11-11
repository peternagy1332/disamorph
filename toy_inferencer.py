import numpy as np
import tensorflow as tf
import math

from model.model_inference import BuildInferenceModel
from model.model_train import BuildTrainModel


class ToyInferencer(object):
    def __init__(self, model_configuration, data_processor):
        self.__config = model_configuration
        self.__data_processor = data_processor

        build_train_model = BuildTrainModel(model_configuration,
                                            self.__data_processor.vocabulary,
                                            self.__data_processor.inverse_vocabulary)

        build_train_model.create_model()

        # Building inference model
        build_inference_model = BuildInferenceModel(model_configuration,
                                                    self.__data_processor.inverse_vocabulary,
                                                    build_train_model)

        self.__inference_model = build_inference_model.create_model()

        self.__session = tf.Session(graph=self.__inference_model.graph)

    def vectorize_input_string(self, input):
        input_words = input.split(' ')

        input_vector = [self.__data_processor.vocabulary.get(word, self.__config.marker_unknown) for word in input_words]
        input_vector += [self.__config.marker_padding] * (self.__config.max_source_sequence_length - len(input_vector))

        return np.matrix(input_vector)

    def unvectorize_network_output(self, network_output):
        network_output_to_list = []
        for row in network_output:
            all_words = [self.__data_processor.inverse_vocabulary.get(word_id, self.__data_processor.inverse_vocabulary[self.__config.marker_unknown]) for word_id in row]
            words = []
            for word in all_words:
                if word != '':
                    words.append(word)

            network_output_to_list.append(words)

        return network_output_to_list


    def inference_batch_to_output(self, inference_batch):

        with self.__inference_model.graph.as_default():
            saver = tf.train.Saver()

            #self.__session.run(tf.tables_initializer())
            #self.__session.run(tf.global_variables_initializer())

            saver.restore(self.__session, self.__config.train_files_save_model)

            final_outputs = self.__session.run(
                [self.__inference_model.final_outputs],
                feed_dict={
                    self.__inference_model.placeholders.infer_inputs: inference_batch
                }
            )

            # print(final_outputs[0].rnn_output[0].shape)
            # print(np.max(final_outputs[0].rnn_output[0], axis=1))
            # print(math.e ** np.sum(np.log(np.max(final_outputs[0].rnn_output[0], axis=1))))

            return final_outputs[0].sample_id


    def evaluate_model(self, source_input_examples, target_output_examples):
        result = self.inference_batch_to_output(source_input_examples)

        output_strings = self.unvectorize_network_output(result)
        target_strings = self.unvectorize_network_output(target_output_examples.tolist())

        print('%-110s%-s' % ("Target output sequences", "Network output sequences"))
        equality_count = 0
        for i in range(len(target_strings)):
            print('%-110s%-s' % (" ".join(target_strings[i]),'|  '+" ".join(output_strings[i])))
            if target_strings[i] == output_strings[i]:
                equality_count+=1

        print('Accuracy: ', 100*equality_count/len(target_strings))