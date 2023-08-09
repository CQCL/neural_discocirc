import tensorflow as tf

from neural.trainers.trainer_base_class import TrainerBaseClass
from neural.utils.utils import get_fast_nn_functor, initialize_boxes


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

    def compile_diagrams(self, diagrams):
        compiled_diagrams = []
        for diag in diagrams:
            compiled_diagrams.append(self.nn_functor(diag))

        return compiled_diagrams


    def train_step(self, batch):
        losses = 0
        grads = None
        for idx in batch:
            loss, grd = self._train_step_for_sample([int(idx.numpy())])
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

    def call(self, circuit):
        assert(len(circuit) == 1)
        return circuit[0](tf.convert_to_tensor([[]]))
