import numpy as np
import tensorflow as tf

from neural.models.model_base_class import ModelBaseClass
from neural.utils.circuit_to_textspace import TextSpace
from neural.utils.utils import create_feedforward_network


class TextspaceModel(ModelBaseClass):
    def __init__(self,
                 wire_dimension,
                 textspace_dimension,
                 latent_dimension,
                 question_length=1,
                 max_wire_num=20,
                 lexicon=None,
                 vocab_dict=None,
                 expansion_hidden_layers=None,
                 contraction_hidden_layers=None,
                 qna_classifier_model=None,
                 qna_hidden_layers=None,
                 circuit_to_textspace=None,
            ):
        super().__init__(wire_dimension=wire_dimension,
                         context_key="context_circ",
                         question_key="question_circ",
                         answer_key="answer")

        if circuit_to_textspace is None:
            self.circuit_to_textspace = TextSpace(
                wire_dim=wire_dimension,
                textspace_dim=textspace_dimension,
                latent_dim=latent_dimension,
                expansion_hidden_layers=expansion_hidden_layers,
                contraction_hidden_layers=contraction_hidden_layers,
            )
        else:
            self.circuit_to_textspace = circuit_to_textspace

        self.max_wire_num = max_wire_num
        self.textspace_dimension = textspace_dimension

        if vocab_dict is None:
            self.vocab_dict = {}
            for v in lexicon:
                if v.name not in self.vocab_dict.keys():
                    self.vocab_dict[v.name] = len(self.vocab_dict)
        else:
            self.vocab_dict = vocab_dict

        if qna_classifier_model is None:
            self.qna_classifier_model = create_feedforward_network(
                input_dim = 2 * self.textspace_dimension,
                output_dim = len(self.vocab_dict),
                hidden_layers = qna_hidden_layers
            )
        else:
            self.qna_classifier_model = qna_classifier_model


    # @tf.function(jit_compile=True)
    def get_answer_prob(self, contexts, questions):
        context_vectors = self.circuit_to_textspace(contexts)

        question_vectors = self.circuit_to_textspace(questions)

        classifier_input = tf.concat([context_vectors, question_vectors], axis=1)
        return self.qna_classifier_model(classifier_input)


    def get_config(self):
        config = super().get_config()
        config.update({
            "qna_classifier_model": self.qna_classifier_model,
            "space_expansion": self.circuit_to_textspace.space_expansion,
            "space_contraction": self.circuit_to_textspace.space_contraction,
            "wire_dimension": self.wire_dimension,
            "max_wire_num": self.max_wire_num,
            "textspace_dimension": self.textspace_dimension,
            "vocab_dict": self.vocab_dict
        })
        return config

    def get_prediction_result(self, call_result):
        return np.argmax(call_result)

    def get_expected_result(self, given_value):
        # TODO: figure out how to handle for tasks 8 and 19
        return self.vocab_dict[given_value[0]]
