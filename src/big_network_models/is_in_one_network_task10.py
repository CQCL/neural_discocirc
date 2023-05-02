import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow import keras

from big_network_models.one_network_trainer_base import OneNetworkTrainerBase
from utils.utils import create_feedforward_network


class IsInOneNetworkTrainerTask10(OneNetworkTrainerBase):
    def __init__(self, is_in_question=None,
                 is_in_hidden_layers=[10], **kwargs):
        super(IsInOneNetworkTrainerTask10, self).__init__(**kwargs)
        self.is_in_question = is_in_question
        wire_dimension = kwargs["wire_dimension"]
        if is_in_question is None:
            self.is_in_question = create_feedforward_network(
                input_dim=2 * wire_dimension,
                output_dim=3,
                hidden_layers=is_in_hidden_layers
            )

    # @tf.function(jit_compile=True)
    def compute_loss(self, outputs, tests):
        answer, answer_prob = self._get_answer_prob(outputs, tests)
        categorical_cross_entropy = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
        answer = np.stack(answer)
        loss = categorical_cross_entropy(answer, answer_prob)
        return loss

    # @tf.function(jit_compile=True)
    def _get_answer_prob(self, outputs, tests):
        num_wires = outputs.shape[1] // self.wire_dimension
        output_wires = tf.split(outputs, num_wires, axis=1)
        tests = np.array(tests).T
        question_tuple, y_n_maybe_answer = tests[0], tests[1]
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

        # print(answer_prob)

        answer_prob = answer_prob[0]

        # if len(answer_prob) > 1:
        #     answer_prob = tf.transpose(answer_prob)

        for i in range(len(y_n_maybe_answer)):
            if y_n_maybe_answer[i] == 'yes':
                y_n_maybe_answer[i] = np.array([1,0,0])
            elif y_n_maybe_answer[i] == 'no':
                y_n_maybe_answer[i] = np.array([0,1,0])
            elif y_n_maybe_answer[i] == 'maybe':
                y_n_maybe_answer[i] = np.array([0,0,1])

        # if y_n_maybe_answer == 'yes':
        #     answer = (1,0,0)
        # elif y_n_maybe_answer == 'no':
        #     answer = (0,1,0)
        # elif y_n_maybe_answer == 'maybe':
        #     answer = (0,0,1)

        return y_n_maybe_answer, answer_prob

    def get_config(self):
        config = super().get_config()
        config.update({
            "is_in_question": self.is_in_question
        })
        return config

    def get_prediction_result(self, call_result):
        return np.argmax(call_result)

    def get_expected_result(self, given_value):
        if given_value == 'yes':
            answer = 0
        elif given_value == 'no':
            answer = 1
        elif given_value == 'maybe':
            answer = 2
        return answer
