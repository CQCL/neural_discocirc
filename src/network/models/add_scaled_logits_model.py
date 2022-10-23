import numpy as np
import tensorflow as tf

from network.models.model_base_class import ModelBaseClass
from network.utils.utils import create_feedforward_network


class AddScaledLogitsModel(ModelBaseClass):
    def __init__(self,
                 wire_dimension,
                 softmax_relevancies,
                 softmax_logits,
                 vocab_dict=None,
                 lexicon=None,
                 relevance_question=None,
                 is_in_question=None,
                 is_in_hidden_layers=None,
                 relevance_hidden_layers=None,
            ):
        super().__init__(wire_dimension=wire_dimension,
                         context_circuit_key="context_circ",
                         question_key="question_id",
                         answer_key="answer")

        self.softmax_relevancies = softmax_relevancies
        self.softmax_logits = softmax_logits

        if vocab_dict is None:
            self.vocab_dict = {}
            for i, v in enumerate(lexicon):
                self.vocab_dict[v.name] = i
        else:
            self.vocab_dict = vocab_dict

        if is_in_question is None:
            self.is_in_question = create_feedforward_network(
                input_dim = 2 * wire_dimension,
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
    def get_answer_prob(self, outputs, person):
        num_wires = outputs.shape[1] // self.wire_dimension
        output_wires = tf.split(outputs, num_wires, axis=1)
        person = [(int(person), i) for i, person in enumerate(person)]
        person_vectors = tf.gather_nd(output_wires, person)

        relevances = []
        for i in range(num_wires):
            location_vectors = output_wires[i]
            relevances.append(tf.squeeze(
                self.relevance_question(
                    tf.concat([person_vectors, location_vectors], axis=1)
                )
            ))

        relevances = tf.convert_to_tensor(relevances)
        if self.softmax_relevancies:
            relevances = tf.nn.softmax(relevances)

        logit_sum = [tf.zeros(len(self.vocab_dict)) for _ in range(len(person))]
        for i in range(num_wires):
            location_vectors = output_wires[i]

            answer = self.is_in_question(
                    tf.concat([person_vectors, location_vectors], axis=1)
            )

            logit = tf.convert_to_tensor(answer)
            if self.softmax_logits:
                logit = tf.nn.softmax(logit)

            for j in range(len(logit)): # loop over each value in the batch (this should probably be done differently)
                relevance = relevances[i]
                if len(relevances.shape) > 1:
                    relevance = relevances[i][j] # the reason why I loop over the batches. Not sure who to do this otherwise
                logit_sum[j] = tf.math.add(
                        tf.math.multiply(logit[j], relevance),
                        logit_sum[j]
                )

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
        return self.vocab_dict[given_value]
