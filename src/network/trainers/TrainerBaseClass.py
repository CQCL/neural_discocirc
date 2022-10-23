from abc import abstractmethod

from tensorflow import keras

class OneNetworkTrainer(keras.Model):
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
    @classmethod
    def load_model_trainer(self):
        pass

    @classmethod
    def load_model(cls, path, model_class):
        model = keras.models.load_model(
            path,
            custom_objects={cls.__name__: cls,
                            model_class.__name__: model_class},
        )
        model.run_eagerly = True

        cls.load_model_trainer()

        return model