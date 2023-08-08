

from typing import Tuple, TypedDict


class NeuralConfig(TypedDict):
    wire_dimension: int
    hidden_layers: Tuple[int, ...]
    is_in_hidden_layers: Tuple[int, ...]
    relevance_hidden_layers: Tuple[int, ...]
    softmax_relevancies: bool
    softmax_logits: bool
    expansion_hidden_layers: Tuple[int, ...]
    contraction_hidden_layers: Tuple[int, ...]
    latent_dimension: int
    textspace_dimension: int
    qna_hidden_layers: Tuple[int, ...]
    lstm_dimension: int



def get_neural_config(**kwargs):
    """Generate a config object with defaults."""
    default_config = NeuralConfig(
        wire_dimension=10,
        hidden_layers=[10, 10],
        is_in_hidden_layers=[10, 10],
        relevance_hidden_layers=[10, 10],
        softmax_relevancies=False,
        softmax_logits=False,
        expansion_hidden_layers=[20, 50],
        contraction_hidden_layers=[50, 20],
        latent_dimension=100,
        textspace_dimension=20,
        qna_hidden_layers=[10, 10],
        lstm_dimension=10,
    )
    for kwarg, val in kwargs.items():
        if kwarg in default_config:
            default_config[kwarg] = val
    return default_config