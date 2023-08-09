from abc import abstractmethod
import pickle
from typing import Tuple

from sklearn.metrics import accuracy_score
from tensorflow import keras
import tensorflow as tf
from shared.config.config import CompilationConfig, TrainerConfig

from shared.trainer.base import Trainer


class TrainerBaseClass(keras.Model, Trainer):
    def __init__(self, config: TrainerConfig):
        super().__init__()
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        self.learning_rate = config['learning_rate']
        
        with open(config['neural']['lexicon_path'], 'rb') as file:
            self.lexicon = pickle.load(file)
            
        self.wire_dimension = config['neural']['wire_dimension']
        self.hidden_layers = list(config['neural']['hidden_layers'])
        self.model_class = config['model_class']
        self.model = self.model_class(config['neural'],
                                       lexicon=self.lexicon)
        self.loss_tracker = keras.metrics.Mean(name="loss")

    def compile_data(self, data: list, config: CompilationConfig, **kwargs) -> list:
        # All diagrams of one key have to be compiled at the same time. Thus we compile per key.
        compiled_data = {}
        for key in [self.model.context_key,
                    self.model.question_key,
                    self.model.answer_key]:
            current_data = [entry[key] for entry in data]
            if key in self.model.data_requiring_compilation:
                compiled_data[key] = self.compile_diagrams(current_data)
            else:
                compiled_data[key] = current_data
                
        # Bring data back into list form
        output_data = []
        for i in range(len(compiled_data[self.model.context_key])):
            output_data.append({
                self.model.context_key: compiled_data[self.model.context_key][i],
                self.model.question_key: compiled_data[self.model.question_key][i],
                self.model.answer_key: compiled_data[self.model.answer_key][i]
            })

        return output_data

    # @tf.function
    def _train_step_for_sample(self, batch_index):
        contexts = [self.dataset[i][self.model.context_key]
                    for i in batch_index]
        questions = [self.dataset[i][self.model.question_key]
                    for i in batch_index]
        answers = [self.dataset[i][self.model.answer_key]
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

    def evaluate(self, data: list, **kwargs) -> Tuple[float, int]:
        location_predicted = []
        location_true = []

        for i in range(len(data[self.model.context_key])):
            print('predicting {} / {}'.format(i, len(data)), end='\r')

            contexts, questions, answers = self.call_on_dataset({
                self.model.context_key: [data[self.model.context_key][i]],
                self.model.question_key: [data[self.model.question_key][i]],
                self.model.answer_key: [data[self.model.answer_key][i]]
            })
            answer_prob = self.model.get_answer_prob(contexts, questions)

            location_predicted.append(
                self.model.get_prediction_result(answer_prob[0])
            )
            location_true.append(
                self.model.get_expected_result(answers[0])
            )

        accuracy = accuracy_score(location_true, location_predicted)
        return accuracy, 0

    def fit(self, data: list, start_epoch=0, epoch_callback=None):
        # TODO: this should not be here
        self.model.build([])
        self.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=0.001),
            run_eagerly=True
        )
    
        self.dataset = data

        input_index_dataset = tf.data.Dataset.range(len(self.dataset))
        input_index_dataset = input_index_dataset.shuffle(len(self.dataset))
        input_index_dataset = input_index_dataset.batch(self.batch_size)

        # TODO: add callbacks
        return super().fit(input_index_dataset, epochs=self.epochs, callbacks=None)