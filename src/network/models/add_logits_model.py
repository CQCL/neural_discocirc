import numpy as np
import tensorflow as tf

from network.models.model_base_class import ModelBaseClass
from network.utils.utils import create_feedforward_network


class AddLogitsModel(ModelBaseClass):
    def __init__(self,
                 wire_dimension,
                 is_in_hidden_layers,
                 softmax_logits,
                 vocab_dict=None,
                 lexicon=None,
                 is_in_question=None,
            ):
        super().__init__(wire_dimension=wire_dimension,
                         context_key="context_circ",
                         question_key="question_id",
                         answer_key="answer")

        self.softmax_logits = softmax_logits

        if vocab_dict is None:
            self.vocab_dict = {}
            for i, v in enumerate(lexicon):
                self.vocab_dict[v.name] = i
        else:
            self.vocab_dict = vocab_dict

        if is_in_question is None:
            self.is_in_question = create_feedforward_network(
                input_dim=2 * wire_dimension,
                output_dim=len(self.vocab_dict),
                hidden_layers=is_in_hidden_layers
            )
        else:
            self.is_in_question = is_in_question

    # @tf.function(jit_compile=True)
    def get_answer_prob(self, contexts, questions):
        num_wires = contexts.shape[1] // self.wire_dimension
        output_wires = tf.split(contexts, num_wires, axis=1)
        questions = [(int(person), i) for i, person in enumerate(questions)]
        person_vectors = tf.gather_nd(output_wires, questions)

        logit_sum = tf.zeros((len(contexts), len(self.vocab_dict)))
        for i in range(num_wires):
            logit = self.is_in_question(
                tf.concat([person_vectors, output_wires[i]], axis=1)
            )

            logit = tf.convert_to_tensor(logit)
            if self.softmax_logits:
                logit = tf.nn.softmax(logit)

            logit_sum = tf.math.add(logit, logit_sum)

        return logit_sum

    def get_config(self):
        config = super().get_config()
        config.update({
            "wire_dimension": self.wire_dimension,
            "is_in_question": self.is_in_question,
            "vocab_dict": self.vocab_dict,
        })
        return config

    def get_prediction_result(self, call_result):
        return np.argmax(call_result)

    def get_expected_result(self, given_value):
        return self.vocab_dict[given_value]
