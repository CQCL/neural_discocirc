from __future__ import annotations
import abc
from typing import Union
from dataclasses import dataclass
from shared.config.utils import WandbProject
from .base import Trainer


class InitParams(abc.ABC):
    """Base class to define strategies for initialising model params"""
    kind_str: str  # root of repr string for this class

    @classmethod
    def from_string(cls, string: str) -> InitParams:
        kind, params = string.split("|")
        assert (kind == cls.kind_str)
        return cls(*params.split(", "))


@dataclass
class SavedParams(InitParams):
    """
    Class to specify where previously saved trainer/model params are saved.
    The specific interface to expose is defined based on the specific trainer used and will depend on the model backend.
    """
    kind_str = "SavedTrainer"

    model_name: str
    use_wandb: bool
    wandb_path: WandbProject
    local_path: str

    def __init__(self, model_name: str, use_wandb: Union[bool, str], wandb_path: Union[WandbProject, str], local_path: str):
        self.model_name = model_name
        self.use_wandb = True if str(use_wandb) == 'True' else False
        self.wandb_path = wandb_path if isinstance(wandb_path, WandbProject) else WandbProject.from_path(wandb_path)
        self.local_path = local_path

    def __repr__(self):
        return f"{self.kind_str}|{self.model_name}, {self.use_wandb}, {self.wandb_path.run_path()}, {self.local_path}"

