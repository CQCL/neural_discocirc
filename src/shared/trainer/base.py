from __future__ import annotations
import abc
from typing import Tuple
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from shared.config import TrainerConfig, CompilationConfig, Device
    from shared.trainer import SavedParams


class Trainer(abc.ABC):
    """
    Trainer interface used by the run and evaluation cycles.
    """

    def __init__(self, data: list, config: TrainerConfig, **kwargs):
        """Init the trainer and model"""

    def compile(self, data: list, config: CompilationConfig, **kwargs) -> list:
        """Compile discopy diagrams into neural networks/quantum circuits."""

    def evaluate(self, data: list, **kwargs) -> Tuple[float, int]:
        """Evaluate the model on the given data.
        Return a pair: (
            accuracy: float,  - the average accuaracy over the supplied dataset
            n_skipped: int    - the number of instances skipped
        )
        """

    def fit(self, data: list, start_epoch=0, epoch_callback=None):
        """
        Training loop.
        @param epoch_callback: function(epoch, loss, gradients), called at the end of each epoch.
        """

    def to(self, device: Device):
        """Convert to a specified device"""

    def save(self, filename: str):
        """Save the trainer and model states for resuming."""

    @classmethod
    def load(cls, saved_params: SavedParams, config: TrainerConfig, data: list, **kwargs) -> Trainer:
        """Load a previously saved trainer."""
