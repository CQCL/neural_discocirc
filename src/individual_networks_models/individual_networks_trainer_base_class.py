import pickle
from abc import ABC, abstractmethod

import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow import keras

from utils.utils import get_fast_nn_functor, initialize_boxes


class IndividualNetworksTrainerBase(ABC, keras.Model):
    def __init__(self, lexicon=None, wire_dimension=10, hidden_layers=[10, 10],
                     **kwargs):
        super(IndividualNetworksTrainerBase, self).__init__(**kwargs)
        if lexicon is not None:
            self.nn_boxes = initialize_boxes(lexicon, wire_dimension, hidden_layers)
            self.nn_functor = get_fast_nn_functor(self.nn_boxes, wire_dimension)
        self.hidden_layers = hidden_layers
        self.wire_dimension = wire_dimension
        self.dataset = None
        self.lexicon = lexicon
        self.loss_tracker = keras.metrics.Mean(name="loss")

    def get_config(self):
        return {
            "wire_dimension": self.wire_dimension,
            "hidden_layers": self.hidden_layers,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @classmethod
    def load_model(cls, path):
        model = keras.models.load_model(
            path,
            custom_objects={cls.__name__: cls},
        )
        model.nn_functor = get_fast_nn_functor(model.nn_boxes, model.wire_dimension)
        model.run_eagerly = True
        return model


    def compile_dataset(self, dataset):
        """
        applies the nn_functor to the list of context circuit diagrams,
        and saves these
        """
        model_dataset = []
        count = 0
        for context_circuit, test in dataset:
            print(count + 1, "/", len(dataset), end="\r")
            count += 1
            context_circuit_model = self.nn_functor(context_circuit)
            model_dataset.append([context_circuit_model, test])

        return model_dataset

    def train_step(self, batch):
        losses = 0
        grads = None
        for idx in batch:
            loss, grd = self.train_step_for_sample(
                self.dataset[int(idx.numpy())])
            losses += loss
            if grads is None:
                grads = grd
            else:
                grads = [g1 + g2 for g1, g2 in zip(grads, grd)]
        grads = [g / len(batch) for g in grads]
        losses = losses / len(batch)
        self.optimizer.apply_gradients((grad, weights)
                                       for (grad, weights) in
                                       zip(grads, self.trainable_weights)
                                       if grad is not None)

        self.loss_tracker.update_state(losses)
        return {
            "loss": self.loss_tracker.result(),
        }

    @tf.function
    def train_step_for_sample(self, dataset):
        with tf.GradientTape() as tape:
            context_circuit_model, test = dataset
            loss = self.compute_loss(context_circuit_model, test)
            grad = tape.gradient(loss, self.trainable_weights,
                                 unconnected_gradients=tf.UnconnectedGradients.ZERO)
        return loss, grad

    @abstractmethod
    def get_prediction_result(self, call_result):
        """
        Given the result of a single call to the network,
        give the prediction of the 

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
        on the output of the 

        :param given_value: The ground truth given in the dataset.
        :return: The expected output of the model.
        """
        pass

    def get_accuracy(self, dataset):
        location_predicted = []
        location_true = []
        for i in range(len(dataset)):
            print('predicting {} / {}'.format(i, len(dataset)), end='\r')

            data = dataset[i]
            probs = self((data[0], data[1][0]))

            location_predicted.append(self.get_prediction_result(probs))
            location_true.append(self.get_expected_result(data[1][1]))

        accuracy = accuracy_score(location_true, location_predicted)
        return accuracy

    @abstractmethod
    def compute_loss(self, context_circuit_model, test):
        pass

    def fit(self, train_dataset, validation_dataset, epochs, batch_size=32, **kwargs):
        print('compiling train dataset (size: {})...'.
              format(len(train_dataset)))

        self.dataset = self.compile_dataset(train_dataset)
        self.dataset_size = len(self.dataset)

        print('compiling validation dataset (size: {})...'
              .format(len(validation_dataset)))
        self.validation_dataset = self.compile_dataset(validation_dataset)


        input_index_dataset = tf.data.Dataset.range(self.dataset_size)
        input_index_dataset = input_index_dataset.shuffle(self.dataset_size)
        input_index_dataset = input_index_dataset.batch(batch_size)

        return super(IndividualNetworksTrainerBase, self).fit(input_index_dataset,
                                                              epochs=epochs, **kwargs)
