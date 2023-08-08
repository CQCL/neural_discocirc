from typing import TypedDict, Optional, Literal, Union, NamedTuple
from dataclasses import dataclass, asdict, fields


SENTINEL = object()  # Allow None values


def dictify(cls):
    """
    decorator to add dict-like methods to a dataclass
    """
    def __getitem__(self, key):
        return getattr(self, key)

    cls.__getitem__ = __getitem__

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    cls.__setitem__ = __setitem__

    def __contains__(self, key):
        return key in asdict(self)

    cls.__contains__ = __contains__

    def from_dict(mapping):
        for field in fields(cls):
            value = mapping.get(field.name, SENTINEL)
            if value != SENTINEL and not isinstance(value, field.type):
                raise TypeError(f"Field ''{field.name}' value '{value}' not of type '{field.type}'.")

        return cls(**mapping)

    cls.from_dict = from_dict

    cls.__len__ = lambda self: len(asdict(self))
    cls.keys = lambda self: asdict(self).keys()
    cls.values = lambda self: asdict(self).values()
    cls.items = lambda self: asdict(self).items()

    return cls


class Device(NamedTuple):
    device_type: Union[Literal['cpu'], Literal['gpu']]
    gpu_no: Optional[int]

    def __repr__(self):
        return (self.device_type, self.gpu_no).__repr__()


@dictify
@dataclass(kw_only=True)
class ValidationConfig:
    # Add validation options
    ref: str = ""  # id for this cross validation sequence
    type: str = "Kfold"  # eg KFold
    n_splits: int = 5  # number of splits to use in the data for KFold validation
    split: Optional[int] = None  # log index of split (out of n_splits)
    n_repeats: int = 5  # number of random initialisations for each fold.
    iteration: Optional[int] = None  # log index of current iteration (out of n_repeats)


@dictify
@dataclass(kw_only=True)
class EnvConfig:
    dataset_path: str = ''  # path to saved datasets. Includes trailing /
    device: Device = Device("cpu", None)  # device to target for training.


class WandbProject(NamedTuple):
    entity: str
    project: str
    run: Optional[str]  # use for resuming

    def __repr__(self):
        return (self.entity, self.project, self.run).__repr__()

    def project_path(self):
        return f"{self.entity}/{self.project}"

    def run_path(self):
        return f"{self.project_path()}/{self.run}"

    @classmethod
    def from_path(cls, path):
        return cls(*path.split('/'))


@dictify
@dataclass(kw_only=True)
class LoggingConfig:
    save_rate: int = 3  # how often to checkpoint the model. shared
    use_wandb: bool = False  # whether to log stuff to wandb. shared.
    wandb_project: Optional[WandbProject] = None  # path to wandb project. shared.
    local_path: Optional[str] = './'  # local path to save files to if not using wandb
