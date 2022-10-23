from discopy.monoidal import Diagram
import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow import keras

from network.utils.utils import get_fast_nn_functor, initialize_boxes


class IndividualNetworksTrainer(keras.Model):
    def __init__(self,
                 lexicon,
                 wire_dimension,
                 hidden_layers,
                 model_class,
                 **kwargs):
        super().__init__()
        self.nn_boxes = initialize_boxes(lexicon, wire_dimension,
                                         hidden_layers)
        self.nn_functor = get_fast_nn_functor(self.nn_boxes, wire_dimension)
        self.hidden_layers = hidden_layers
        self.wire_dimension = wire_dimension
        self.lexicon = lexicon
        self.model_class = model_class(wire_dimension=wire_dimension,
                                       lexicon=lexicon, **kwargs)
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
    def load_model(cls, path, model_class):
        model = keras.models.load_model(
            path,
            custom_objects={cls.__name__: cls,
                            model_class.__name__: model_class},
        )
        model.nn_functor = get_fast_nn_functor(model.nn_boxes,
                                               model.wire_dimension)
        model.run_eagerly = True
        return model

    def compile_dataset(self, dataset):
        """
        applies the nn_functor to the list of context circuit diagrams,
        and saves these
        """
        model_dataset = []
        count = 0
        for data in dataset:
            print(count + 1, "/", len(dataset), end="\r")
            count += 1

            compiled_data = {}
            for key, result in [(self.model_class.context_circuit_key, "context"),
                                (self.model_class.question_key, "question"),
                                (self.model_class.answer_key, "answer")]:
                if key in self.model_class.data_requiring_compilation:
                    compiled_data[result] = self.nn_functor(data[key])
                else:
                    compiled_data[result] = data[key]

            model_dataset.append([
                compiled_data["context"],
                (compiled_data["question"], compiled_data["answer"])
            ])

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
            output_vector = self.call(context_circuit_model)
            loss = self.model_class.compute_loss(output_vector, [test])
            grad = tape.gradient(loss, self.trainable_weights,
                                 unconnected_gradients=tf.UnconnectedGradients.ZERO)
        return loss, grad

    def call(self, circuit):
        return circuit(tf.convert_to_tensor([[]]))

    def get_accuracy(self, dataset):
        location_predicted = []
        location_true = []
        # if type(dataset[0][0]) == Diagram:
        # dataset = self.compile_dataset(dataset)

        for i in range(len(dataset)):
            print('predicting {} / {}'.format(i, len(dataset)), end='\r')

            data = dataset[i]
            outputs = self.call(data[0])
            answer_prob = self.model_class.get_answer_prob(outputs,
                                                              [data[1][0]])

            location_predicted.append(
                self.model_class.get_prediction_result(answer_prob)
            )
            location_true.append(
                self.model_class.get_expected_result(data[1][1])
            )

        accuracy = accuracy_score(location_true, location_predicted)
        return accuracy

    def fit(self, train_dataset, validation_dataset, epochs, batch_size=32,
            **kwargs):
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

        return super(IndividualNetworksTrainer, self).fit(input_index_dataset,
                                                          epochs=epochs,
                                                          **kwargs)
