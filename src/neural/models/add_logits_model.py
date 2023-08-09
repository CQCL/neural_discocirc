import numpy as np
import tensorflow as tf

from neural.models.model_base_class import ModelBaseClass
from neural.utils.utils import create_feedforward_network
from shared.config.neural_config import NeuralConfig


class AddLogitsModel(ModelBaseClass):
    def __init__(self,
                 config:NeuralConfig,
                 vocab_dict=None,
                 lexicon=None,
                 is_in_question=None,
            ):
        super().__init__(wire_dimension=config["wire_dimension"],
                         context_key="context_circ",
                         question_key="question_id",
                         answer_key="answer")

        self.softmax_logits = config["softmax_logits"]

        if vocab_dict is None:
            self.vocab_dict = {}
            for v in lexicon:
                if v.name not in self.vocab_dict.keys():
                    self.vocab_dict[v.name] = len(self.vocab_dict)
        else:
            self.vocab_dict = vocab_dict

        if is_in_question is None:
            self.is_in_question = create_feedforward_network(
                input_dim=(config["question_length"] + 1) * config["wire_dimension"],
                output_dim=len(self.vocab_dict),
                hidden_layers=config["is_in_hidden_layers"],
            )
        else:
            self.is_in_question = is_in_question

    # @tf.function(jit_compile=True)
    def get_answer_prob(self, contexts, questions):
        num_wires = contexts.shape[1] // self.wire_dimension
        output_wires = tf.split(contexts, num_wires, axis=1)
        questions = [
            [(int(question[j]), i) for i, question in enumerate(questions)]
            for j in range(len(questions[0]))]
        question_vector = tf.concat([tf.gather_nd(output_wires, question) for question in questions], axis=1)

        logit_sum = tf.zeros((len(contexts), len(self.vocab_dict)))
        for i in range(num_wires):
            logit = self.is_in_question(
                tf.concat([question_vector, output_wires[i]], axis=1)
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
        # TODO: figure out how to handle for tasks 8 and 19
        return self.vocab_dict[given_value[0]]