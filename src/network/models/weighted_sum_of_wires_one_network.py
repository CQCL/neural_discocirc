import numpy as np
import tensorflow as tf

from network.models.model_base_class import ModelBaseClass
from utils.utils import create_feedforward_network


class WeightedSumOfWiresModel(ModelBaseClass):
    def __init__(self,
                 wire_dimension,
                 vocab_dict=None,
                 lexicon=None,
                 relevance_question=None,
                 is_in_question=None,
                 is_in_hidden_layers=None,
                 relevance_hidden_layers=None,
            ):
        super().__init__(wire_dimension=wire_dimension,
                         context_key="context_circ",
                         question_key="question_id",
                         answer_key="answer")

        if vocab_dict is None:
            self.vocab_dict = {}
            for i, v in enumerate(lexicon):
                self.vocab_dict[v.name] = i
        else:
            self.vocab_dict = vocab_dict

        if is_in_question is None:
            self.is_in_question = create_feedforward_network(
                input_dim = wire_dimension,
                output_dim = len(self.vocab_dict),
                hidden_layers = is_in_hidden_layers
            )
        else:
            self.is_in_question = is_in_question

        if relevance_question is None:
            self.relevance_question = create_feedforward_network(
                input_dim = 2 * wire_dimension,
                output_dim = 1,
                hidden_layers = relevance_hidden_layers
            )
        else:
            self.relevance_question = relevance_question

    # @tf.function(jit_compile=True)
    def get_answer_prob(self, contexts, questions):
        num_wires = contexts.shape[1] // self.wire_dimension
        output_wires = tf.split(contexts, num_wires, axis=1)
        questions = [(int(person), i) for i, person in enumerate(questions)]
        person_vectors = tf.gather_nd(output_wires, questions)

        wire_sum = tf.zeros((len(contexts), self.wire_dimension))
        for i in range(num_wires):
            relevances = tf.squeeze(self.relevance_question(
                    tf.concat([person_vectors, output_wires[i]], axis=1)
            ), axis=1)
            wire_sum = wire_sum + tf.einsum("ij,i->ij", output_wires[i],
                                            relevances)

        logit = self.is_in_question(wire_sum)
        return logit

    def get_config(self):
        config = super().get_config()
        config.update({
            "wire_dimension": self.wire_dimension,
            "is_in_question": self.is_in_question,
            "relevance_question": self.relevance_question,
            "vocab_dict": self.vocab_dict,
        })
        return config

    def get_prediction_result(self, call_result):
        return np.argmax(call_result)

    def get_expected_result(self, given_value):
        return self.vocab_dict[given_value]
