import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow import keras

from big_network_models.one_network_trainer_base import OneNetworkTrainerBase
from utils.utils import create_feedforward_network_binary


class IsInOneNetworkTrainerTask6(OneNetworkTrainerBase):
    def __init__(self, is_in_question=None,
                 is_in_hidden_layers=[10], **kwargs):
        super(IsInOneNetworkTrainerTask6, self).__init__(**kwargs)
        self.is_in_question = is_in_question
        wire_dimension = kwargs["wire_dimension"]
        if is_in_question is None:
            self.is_in_question = create_feedforward_network_binary(
                input_dim=2 * wire_dimension,
                output_dim=1,
                hidden_layers=is_in_hidden_layers
            )

    # @tf.function(jit_compile=True)
    def compute_loss(self, outputs, tests):
        y_n_answer, answer_prob = self._get_answer_prob(outputs, tests)
        # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=answer_prob, labels=location)
        bce = tf.keras.losses.BinaryCrossentropy(
            from_logits=False, reduction=tf.keras.losses.Reduction.NONE)
        # label_smoothing = 0.0,
        # axis = -1,
        # reduction = losses_utils.ReductionV2.AUTO,
        # name = 'binary_crossentropy'

        #y_n_answer = (np.expand_dims(y_n_answer, axis=1)).tolist()
        #answer_prob = (answer_prob.numpy()).tolist()
        y_n_answer = np.expand_dims(tf.convert_to_tensor(y_n_answer, dtype=tf.float32), axis = 1)
        loss = bce(y_n_answer, answer_prob)
        # loss = bce(y_n_answer, answer_prob).numpy()
        # loss = tf.convert_to_tensor(loss)
        return loss


    # @tf.function(jit_compile=True)
    def _get_answer_prob(self, outputs, tests):
        num_wires = outputs.shape[1] // self.wire_dimension
        output_wires = tf.split(outputs, num_wires, axis=1)
        tests = np.array(tests).T
        question_tuple, y_n_answer = tests[0], tests[1]
        question_tuple = question_tuple.tolist()
        question_tuple = np.array(question_tuple).T
        person, location = question_tuple[0], question_tuple[1]
        person = [(person, i) for i, person in enumerate(person)]
        location = [(location, i) for i, location in enumerate(location)]
        person_vectors = tf.gather_nd(output_wires, person)
        location_vectors = tf.gather_nd(output_wires, location)
        answer_prob = []
        answer_prob.append(tf.squeeze(
            self.is_in_question(
                tf.concat([person_vectors, location_vectors], axis=1)
            )
        ))

        answer_prob = tf.transpose(answer_prob)

        return y_n_answer, answer_prob

    def get_config(self):
        config = super().get_config()
        config.update({
            "is_in_question": self.is_in_question
        })
        return config

    def get_prediction_result(self, call_result):
        return np.where(call_result > 0.5, 1, 0)

    def get_expected_result(self, given_value):
        return given_value
