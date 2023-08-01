import numpy as np
import tensorflow as tf

from models.model_base_class import ModelBaseClass
from utils.utils import create_feedforward_network


class AddScaledLogitsModel(ModelBaseClass):
    def __init__(self,
                 wire_dimension,
                 softmax_relevancies,
                 softmax_logits,
                 question_length=1,
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

        self.softmax_relevancies = softmax_relevancies
        self.softmax_logits = softmax_logits

        if vocab_dict is None:
            self.vocab_dict = {}
            for v in lexicon:
                if v.name not in self.vocab_dict.keys():
                    self.vocab_dict[v.name] = len(self.vocab_dict)
        else:
            self.vocab_dict = vocab_dict

        if is_in_question is None:
            self.is_in_question = create_feedforward_network(
                input_dim = (question_length + 1) * wire_dimension,
                output_dim = len(self.vocab_dict),
                hidden_layers = is_in_hidden_layers
            )
        else:
            self.is_in_question = is_in_question

        if relevance_question is None:
            self.relevance_question = create_feedforward_network(
                input_dim = (question_length + 1) * wire_dimension,
                output_dim = 1,
                hidden_layers = relevance_hidden_layers
            )
        else:
            self.relevance_question = relevance_question

    # @tf.function(jit_compile=True)
    def get_answer_prob(self, contexts, questions):
        num_wires = contexts.shape[1] // self.wire_dimension
        output_wires = tf.split(contexts, num_wires, axis=1)
        questions = [
            [(int(question[j]), i) for i, question in enumerate(questions)]
            for j in range(len(questions[0]))]
        question_vector = tf.concat([tf.gather_nd(output_wires, question) for question in questions], axis=1)

        # TODO: stack for efficiency
        # stack = tf.concat([tf.concat([person_vectors, output_wires[i]], axis=1) for i in range(num_wires)], axis=0)
        # relevance = self.relevance_question(stack)

        relevance = []
        for i in range(num_wires):
            relevance.append(tf.squeeze(
                self.relevance_question(
                    tf.concat([question_vector, output_wires[i]], axis=1)
                ), axis=1
            ))

        relevance = tf.convert_to_tensor(relevance)
        if self.softmax_relevancies:
            relevance = tf.nn.softmax(relevance)

        logit_sum = tf.zeros((len(contexts), len(self.vocab_dict)))
        for i in range(num_wires):
            logits = self.is_in_question(
                tf.concat([question_vector, output_wires[i]], axis=1)
            )
            logits = tf.convert_to_tensor(logits)
            if self.softmax_logits:
                logits = tf.nn.softmax(logits)

            logit_sum = logit_sum + tf.einsum("ij,i->ij", logits, relevance[i])

        return logit_sum

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
        # TODO: figure out how to handle for tasks 8 and 19
        return self.vocab_dict[given_value[0]]
