from abc import abstractmethod

from sklearn.metrics import accuracy_score
from tensorflow import keras
import tensorflow as tf


class TrainerBaseClass(keras.Model):
    def __init__(self,
                 wire_dimension,
                 lexicon,
                 hidden_layers,
                 model_class,
                 **kwargs
                 ):
        super().__init__()
        self.wire_dimension = wire_dimension
        self.hidden_layers = hidden_layers
        self.lexicon = lexicon
        self.model_class = model_class(wire_dimension=wire_dimension,
                                       lexicon=lexicon, **kwargs)
        self.loss_tracker = keras.metrics.Mean(name="loss")

    def compile_dataset(self, dataset):
        compiled_data = {}
        for key in [self.model_class.context_key,
                    self.model_class.question_key,
                    self.model_class.answer_key]:
            current_data = [data[key] for data in dataset]
            if key in self.model_class.data_requiring_compilation:
                diagram_parameters = self.compile_diagrams(current_data)
                compiled_data[key] = diagram_parameters
            else:
                compiled_data[key] = current_data

        return compiled_data

    # @tf.function
    def train_step_for_sample(self, batch_index):
        contexts = [self.dataset[self.model_class.context_key][i]
                    for i in batch_index]
        questions = [self.dataset[self.model_class.question_key][i]
                    for i in batch_index]
        answers = [self.dataset[self.model_class.answer_key][i]
                    for i in batch_index]
        with tf.GradientTape() as tape:
            context_output, question_output, answer_output = \
                self.call_on_dataset({
                    self.model_class.context_key: contexts,
                    self.model_class.question_key: questions,
                    self.model_class.answer_key: answers
                })
            loss = self.model_class.compute_loss(
                context_output, question_output, answer_output)
            grad = tape.gradient(loss, self.trainable_weights,
                                 unconnected_gradients=tf.UnconnectedGradients.ZERO)
        return loss, grad

    def get_config(self):
        return {
            "wire_dimension": self.wire_dimension,
            "hidden_layers": self.hidden_layers,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @abstractmethod
    def load_model_trainer(model):
        pass

    @classmethod
    def load_model(cls, path, model_class):
        model = keras.models.load_model(
            path,
            custom_objects={cls.__name__: cls,
                            model_class.__name__: model_class},
        )
        model.run_eagerly = True
        model = cls.load_model_trainer(model)
        return model

    def call_on_dataset(self, dataset):
        called_data = {}
        for key in [self.model_class.context_key,
                              self.model_class.question_key,
                              self.model_class.answer_key]:
            if key in self.model_class.data_requiring_compilation:
                called_data[key] = self.call(dataset[key])
            else:
                called_data[key] = dataset[key]

        return called_data[self.model_class.context_key], called_data[self.model_class.question_key], called_data[self.model_class.answer_key]

    def get_accuracy(self, dataset):
        location_predicted = []
        location_true = []

        for i in range(len(dataset[self.model_class.context_key])):
            print('predicting {} / {}'.format(i, len(dataset)), end='\r')

            contexts, questions, answers = self.call_on_dataset({
                self.model_class.context_key: [dataset[self.model_class.context_key][i]],
                self.model_class.question_key: [dataset[self.model_class.question_key][i]],
                self.model_class.answer_key: [dataset[self.model_class.answer_key][i]]
            })
            answer_prob = self.model_class.get_answer_prob(contexts,
                                                           questions)

            location_predicted.append(
                self.model_class.get_prediction_result(answer_prob[0])
            )
            location_true.append(
                self.model_class.get_expected_result(answers[0])
            )

        accuracy = accuracy_score(location_true, location_predicted)
        return accuracy

    def fit(self, train_dataset, validation_dataset, epochs, batch_size=32,
            **kwargs):
        print('compiling train dataset (size: {})...'.
              format(len(train_dataset)))

        self.dataset = self.compile_dataset(train_dataset)

        print('compiling validation dataset (size: {})...'
              .format(len(validation_dataset)))
        self.validation_dataset = self.compile_dataset(validation_dataset)

        input_index_dataset = tf.data.Dataset.range(len(train_dataset))
        input_index_dataset = input_index_dataset.shuffle(len(train_dataset))
        input_index_dataset = input_index_dataset.batch(batch_size)

        return super().fit(input_index_dataset, epochs=epochs, **kwargs)