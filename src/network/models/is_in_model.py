import numpy as np
import tensorflow as tf

from network.models.model_base_class import ModelBaseClass
from network.utils.utils import create_feedforward_network


class IsInModel(ModelBaseClass):
    def __init__(self,
                 wire_dimension,
                 is_in_hidden_layers,
                 is_in_question=None,
                 lexicon=None  # not used but always passed by trainer
                 ):
        super().__init__(wire_dimension=wire_dimension,
                         context_key="context_circ",
                         question_key="question_id",
                         answer_key="answer_id"
            )

        if is_in_question is None:
            self.is_in_question = create_feedforward_network(
                input_dim=2 * wire_dimension,
                output_dim=1,
                hidden_layers=is_in_hidden_layers
            )
        else:
            self.is_in_question = is_in_question

    # @tf.function(jit_compile=True)
    def get_answer_prob(self, contexts, questions):
        num_wires = len(contexts[0]) // self.wire_dimension
        output_wires = tf.split(contexts, num_wires, axis=1)

        questions = [(person, i) for i, person in enumerate(questions)]
        person_vectors = tf.gather_nd(output_wires, questions)
        answer_prob = []
        for i in range(num_wires):
            answer_prob.append(tf.squeeze(
                self.is_in_question(
                    tf.concat([person_vectors, output_wires[i]], axis=1)
                )
                , axis=1))
        answer_prob = tf.transpose(answer_prob)
        return answer_prob

    def get_config(self):
        return {
            "wire_dimension": self.wire_dimension,
            "is_in_question": self.is_in_question,
        }

    def get_prediction_result(self, call_result):
        return np.argmax(call_result)

    def get_expected_result(self, given_value):
        return given_value
