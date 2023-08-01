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
        # self.model_class = model_class
        self.model = model_class(wire_dimension=wire_dimension,
                                       lexicon=lexicon, **kwargs)
        self.model_kwargs = kwargs
        self.loss_tracker = keras.metrics.Mean(name="loss")

    def compile_dataset(self, dataset):
        compiled_data = {}
        for key in [self.model.context_key,
                    self.model.question_key,
                    self.model.answer_key]:
            current_data = [data[key] for data in dataset]
            if key in self.model.data_requiring_compilation:
                compiled_data[key] = self.compile_diagrams(current_data)
            else:
                compiled_data[key] = current_data

        return compiled_data

    # @tf.function
    def train_step_for_sample(self, batch_index):
        contexts = [self.dataset[self.model.context_key][i]
                    for i in batch_index]
        questions = [self.dataset[self.model.question_key][i]
                    for i in batch_index]
        answers = [self.dataset[self.model.answer_key][i]
                    for i in batch_index]
        with tf.GradientTape() as tape:
            context_output, question_output, answer_output = \
                self.call_on_dataset({
                    self.model.context_key: contexts,
                    self.model.question_key: questions,
                    self.model.answer_key: answers
                })
            loss = self.model.compute_loss(
                context_output, question_output, answer_output)
            grad = tape.gradient(loss, self.trainable_weights,
                                 unconnected_gradients=tf.UnconnectedGradients.ZERO)
        return loss, grad

    def get_config(self):
        return {
            # "lexicon": self.lexicon, # It can't save the discopy boxes. We should make the key of the lexicon strings of the boxes.
            "wire_dimension": self.wire_dimension,
            "hidden_layers": self.hidden_layers,
            # "model_class": self.model_class, # The problem here is that it is trying to save the class type. But the class has the function get_config. It can be fixed by making this parameter, which is passed to the trainer a string.
            **self.model_kwargs
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @abstractmethod
    def load_model_trainer(self):
        pass

    @classmethod
    def load_model(cls, path, model_class):
        print('loading model from {}'.format(path))
        model = keras.models.load_model(
            path,
            custom_objects={cls.__name__: cls,
                            model_class.__name__: model_class},
        )
        model.run_eagerly = True
        model = model.load_model_trainer()
        return model

    def call_on_dataset(self, dataset):
        called_data = {}
        for key in [self.model.context_key,
                              self.model.question_key,
                              self.model.answer_key]:
            if key in self.model.data_requiring_compilation:
                called_data[key] = self.call(dataset[key])
            else:
                called_data[key] = dataset[key]

        return called_data[self.model.context_key], called_data[self.model.question_key], called_data[self.model.answer_key]

    def get_accuracy(self, dataset):
        location_predicted = []
        location_true = []

        for i in range(len(dataset[self.model.context_key])):
            print('predicting {} / {}'.format(i, len(dataset)), end='\r')

            contexts, questions, answers = self.call_on_dataset({
                self.model.context_key: [dataset[self.model.context_key][i]],
                self.model.question_key: [dataset[self.model.question_key][i]],
                self.model.answer_key: [dataset[self.model.answer_key][i]]
            })
            answer_prob = self.model.get_answer_prob(contexts,
                                                           questions)

            location_predicted.append(
                self.model.get_prediction_result(answer_prob[0])
            )
            location_true.append(
                self.model.get_expected_result(answers[0])
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