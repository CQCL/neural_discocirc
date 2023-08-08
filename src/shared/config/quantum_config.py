from typing import Optional, Tuple, Union
from quantum.ansatz import Ansatz
from dataclasses import dataclass
from .utils import dictify


@dictify
@dataclass(kw_only=True)
class LossConfig:
    type: str = "cross_entropy"  # cross_entropy | clifford ...
    weight: Optional[float] = None
    decay_base: Optional[float] = None
    decay_rate: Optional[int] = None  # number of epochs before we decay the loss function


@dictify
@dataclass(kw_only=True)
class OptimizerConfig:
    type: str = "adam"  # adam | simulated_annealing | spsa
    # Simulated Annealing
    temperature: Optional[float] = None  # initial temperature
    accept_exp: Optional[float] = None  # acceptance exponent modifier.
    perturbations: Optional[float] = None  # proportion of params to perturb
    # SPSA
    a: Optional[Tuple[Union[float, str], Union[float, str], float]] = None  #: a_init, a_decay, a_power.
    c: Optional[Tuple[Union[float, str], float]] = None  #: c_init, c_power.
    delta: Optional[float] = None  # if infering a_init, the average magnitude of the initial parameter update


@dictify
@dataclass(kw_only=True)
class QuantumConfig:
    discopy_backend: str = "torch"  # discopy backend to use for evaluation
    mixed: Optional[bool] = None  # if not mixed, move any discards to the end of the circuit and remove them.

    # bonus config
    loss: LossConfig = LossConfig()  # Define loss type and any required params
    optimizer: OptimizerConfig = OptimizerConfig()  # Define optimiser

    # Ansatz
    ansatz: Ansatz = Ansatz.Sim9Cz
    layers: int = 1
    wire_dimension: int = 1  # dimension for noun wires
    ho_wire_dimension: int = 1  # dimension for the connecting wire on higher order boxes
    discard_ancillas: Optional[bool] = None  # Whether to discard or postselect ancilla wires


def get_quantum_config(**kwargs):
    """Generate a config object with defaults."""
    default_config = QuantumConfig(
        ansatz=Ansatz.Sim9Cz,
        layers=1,
        wire_dimension=1,
        ho_wire_dimension=1,
        discard_ancillas=None,
        loss=LossConfig(type="cross_entropy"),
        optimizer=OptimizerConfig(type="adam"),
        mixed=None,
        discopy_backend='torch',
    )
    for kwarg, val in kwargs.items():
        if kwarg in default_config:
            # Ensure complex types are mapped correctly
            if kwarg == "ansatz" and not isinstance(val, Ansatz):
                val = Ansatz[str(val).split(".")[-1]]
            if val is None and kwarg in ["loss", "optimizer"]:
                continue

            default_config[kwarg] = val
    return default_config
