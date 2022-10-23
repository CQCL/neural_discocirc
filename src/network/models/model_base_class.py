from abc import abstractmethod, ABC

import numpy as np
import tensorflow as tf
from tensorflow import keras


class ModelBaseClass(keras.layers.Layer, ABC):
    data_requiring_compilation = {"context_circ", "question_circ"}

    def __init__(self, wire_dimension, context_circuit_key, question_key, answer_key):
        super().__init__()
        self.wire_dimension = wire_dimension
        self.context_circuit_key = context_circuit_key
        self.question_key = question_key
        self.answer_key = answer_key

    @abstractmethod
    def get_answer_prob(self, outputs, tests):
        pass

    # @tf.function(jit_compile=True)
    def compute_loss(self, outputs, tests):
        tests = np.array(tests).T

        # TODO: if we print test (or anything), the InidivualNetworkTrainer will actually not print during training. How come?
        # print(tests)
        answer_prob = self.get_answer_prob(outputs, tests[0])
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=answer_prob,
            labels=[self.get_expected_result(location)
                        for location in tests[1]]
        )
        return loss

    @abstractmethod
    def get_prediction_result(self, call_result):
        """
        Given the result of a single call to the network,
        give the prediction of the network.
        :param call_result: The results from self.call(...)
        :return: The prediction of the model,
            i.e. the number of the correct wire or the index of the correct word.
        """
        pass

    @abstractmethod
    def get_expected_result(self, given_value):
        """
        Given the ground truth in the dataset, translate into value that model
        should predict after calling get_prediction_result()
        on the output of the network.
        :param given_value: The ground truth given in the dataset.
        :return: The expected output of the model.
        """
        pass

