from abc import abstractmethod

from sklearn.metrics import accuracy_score
from tensorflow import keras

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

    def get_accuracy(self, dataset):
        location_predicted = []
        location_true = []

        for i in range(len(dataset)):
            print('predicting {} / {}'.format(i, len(dataset)), end='\r')

            data = dataset[i]
            outputs = self.call(data[0])
            answer_prob = self.model_class.get_answer_prob(outputs,
                                                           [data[1][0]])

            location_predicted.append(
                self.model_class.get_prediction_result(answer_prob)
            )
            location_true.append(
                self.model_class.get_expected_result(data[1][1])
            )

        accuracy = accuracy_score(location_true, location_predicted)
        return accuracy