import numpy as np
import tensorflow as tf

from network.models.model_base_class import ModelBaseClass
from utils.utils import create_feedforward_network


class IsInMaxWireModel(ModelBaseClass):
    def __init__(self,
                 wire_dimension,
                 is_in_hidden_layers,
                 question_length=1,
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
                input_dim=(question_length + 1) * wire_dimension,
                output_dim=1,
                hidden_layers=is_in_hidden_layers
            )
        else:
            self.is_in_question = is_in_question

    # @tf.function(jit_compile=True)
    def get_answer_prob(self, contexts, questions):
        num_wires = len(contexts[0]) // self.wire_dimension
        output_wires = tf.split(contexts, num_wires, axis=1)

        questions = [
            [(int(question[j]), i) for i, question in enumerate(questions)]
            for j in range(len(questions[0]))]
        question_vector = tf.concat(
            [tf.gather_nd(output_wires, question) for question in questions],
            axis=1)

        answer_prob = []
        for i in range(num_wires):
            answer_prob.append(tf.squeeze(
                self.is_in_question(
                    tf.concat([question_vector, output_wires[i]], axis=1)
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
        # TODO: figure out how to handle for tasks 8 and 19
        return given_value[0]
