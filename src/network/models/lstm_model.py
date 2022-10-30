import keras
import numpy as np
import tensorflow as tf

from network.models.model_base_class import ModelBaseClass

class LSTMModel(ModelBaseClass):
    def __init__(self,
                 wire_dimension,
                 lstm_dimension,
                 lexicon=None,
                 vocab_dict=None,
                 lstm_model=None,
                 classifier_model=None,
            ):
        super().__init__(wire_dimension=wire_dimension,
                         context_key="context_circ",
                         question_key="question_circ",
                         answer_key="answer")

        self.lstm_dimension = lstm_dimension
        self.vocab_dict = vocab_dict
        self.lstm_model = lstm_model
        self.classifier_model = classifier_model
        self.lexicon = lexicon

        if vocab_dict is None:
            self.vocab_dict = {}
            for i, v in enumerate(lexicon):
                self.vocab_dict[v.name] = i
        else:
            self.vocab_dict = vocab_dict

        if self.lstm_model is None:
            self.lstm_model = keras.layers.LSTM(lstm_dimension)

        if self.classifier_model is None:
            self.classifier_model = keras.layers.Dense(len(self.vocab_dict), activation="softmax")



    # @tf.function(jit_compile=True)
    def get_answer_prob(self, contexts, questions):
        num_wires_context = contexts.shape[1] // self.wire_dimension
        output_wires_context = tf.split(contexts, num_wires_context,
                                        axis=1)
        num_wires_question = questions.shape[1] // self.wire_dimension
        output_wires_question = tf.split(questions, num_wires_question,
                                         axis=1)
        contexts = output_wires_context + output_wires_question
        contexts = tf.stack(contexts, axis=1)
        contexts = self.lstm_model(contexts)
        contexts = self.classifier_model(contexts)
        return contexts


    def get_config(self):
        config = super().get_config()
        config.update({
            "lstm_dimension": self.lstm_dimension,
            "wire_dimension": self.wire_dimension,
            "vocab_dict": self.vocab_dict,
            "lstm_model": self.lstm_model,
            "classifier_model": self.classifier_model
        })
        return config

    def get_prediction_result(self, model_output):
        return np.argmax(model_output)

    def get_expected_result(self, given_value):
        return self.vocab_dict[given_value]
