from discopy.monoidal import Diagram
import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow import keras

from network.trainers.trainer_base_class import TrainerBaseClass
from network.utils.utils import get_fast_nn_functor, initialize_boxes


class IndividualNetworksTrainer(TrainerBaseClass):
    def __init__(self,
                 lexicon,
                 wire_dimension,
                 hidden_layers,
                 model_class,
                 **kwargs):
        super().__init__(wire_dimension=wire_dimension, lexicon=lexicon, hidden_layers=hidden_layers, model_class=model_class, **kwargs)
        self.nn_boxes = initialize_boxes(lexicon, wire_dimension,
                                         hidden_layers)
        self.nn_functor = get_fast_nn_functor(self.nn_boxes, wire_dimension)

    @classmethod
    def load_model_tainer(model):
        model.nn_functor = get_fast_nn_functor(model.nn_boxes,
                                               model.wire_dimension)
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
            for key, result in [(self.model_class.context_key, "context"),
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

        self.optimizer.apply_gradients(
            (grad, weights)
            for (grad, weights) in zip(grads, self.trainable_weights)
            if grad is not None)

        self.loss_tracker.update_state(losses)
        return {
            "loss": self.loss_tracker.result(),
        }

    @tf.function
    def train_step_for_sample(self, dataset):
        with tf.GradientTape() as tape:
            context_circuit_model, test = dataset
            output_vector = self.call([context_circuit_model])
            loss = self.model_class.compute_loss(output_vector, [test])
            grad = tape.gradient(loss, self.trainable_weights,
                                 unconnected_gradients=tf.UnconnectedGradients.ZERO)
        return loss, grad

    def call(self, circuit):
        return circuit[0](tf.convert_to_tensor([[]]))
