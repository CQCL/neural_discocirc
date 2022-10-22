from abc import abstractmethod, ABC

from tensorflow import keras


class ModelBaseClass(keras.layers.Layer, ABC):
    def __init__(self, wire_dimension):
        super().__init__()
        self.wire_dimension = wire_dimension

    @abstractmethod
    def get_answer_prob(self, outputs, tests):
        pass

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

