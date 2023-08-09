

from dataclasses import dataclass
from typing import Tuple

from .utils import dictify

@dictify
@dataclass(kw_only=True)
class NeuralConfig:
    wire_dimension: int = 10
    hidden_layers: Tuple[int, ...] = (10, 10)
    is_in_hidden_layers: Tuple[int, ...] = (10, 10)
    relevance_hidden_layers: Tuple[int, ...] = (10, 10)
    softmax_relevancies: bool = False
    softmax_logits: bool = False
    expansion_hidden_layers: Tuple[int, ...] = (20, 50)
    contraction_hidden_layers: Tuple[int, ...] = (50, 20)
    latent_dimension: int = 100
    textspace_dimension: int = 20
    qna_hidden_layers: Tuple[int, ...] = (10, 10)
    lstm_dimension: int = 10
    question_length: int = 1 #TODO: this should not be here
    lexicon_path: str = "data/task_vocab_dicts/task01_train.p" #TODO: this should not be here