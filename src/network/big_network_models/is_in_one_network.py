import numpy as np
import tensorflow as tf
from tensorflow import keras

from network.utils.utils import create_feedforward_network


class IsInOneNetworkTrainer(keras.layers.Layer):
    def __init__(self,
                 wire_dimension,
                 is_in_question=None,
                 is_in_hidden_layers=[10],
                 **kwargs):
        super().__init__()
        self.is_in_question = is_in_question
        self.wire_dimension = wire_dimension
        if is_in_question is None:
            self.is_in_question = create_feedforward_network(
                input_dim = 2 * wire_dimension,
                output_dim = 1,
                hidden_layers = is_in_hidden_layers
            )

    # @tf.function(jit_compile=True)
    def compute_loss(self, outputs, tests):
        location, answer_prob = self._get_answer_prob(outputs, tests)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=answer_prob, labels=location)
        return loss

    # @tf.function(jit_compile=True)
    def _get_answer_prob(self, outputs, tests):
        num_wires = outputs.shape[1] // self.wire_dimension
        output_wires = tf.split(outputs, num_wires, axis=1)
        tests = np.array(tests).T
        person, location = tests[0], tests[1]
        person = [(person, i) for i, person in enumerate(person)]
        person_vectors = tf.gather_nd(output_wires, person)
        answer_prob = []
        for i in range(num_wires):
            location_vectors = output_wires[i]
            answer_prob.append(tf.squeeze(
                self.is_in_question(
                    tf.concat([person_vectors, location_vectors], axis=1)
                )
            , axis=1))
        answer_prob = tf.transpose(answer_prob)
        return location, answer_prob

    def get_config(self):
        return {
            "wire_dimension": self.wire_dimension,
            "is_in_question": self.is_in_question,
        }

    def get_prediction_result(self, call_result):
        return np.argmax(call_result)

    def get_expected_result(self, given_value):
        return given_value
