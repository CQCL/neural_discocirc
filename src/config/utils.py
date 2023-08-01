from typing import TypedDict, Optional, Literal, Union, NamedTuple


class Device(NamedTuple):
    device_type: Union[Literal['cpu'], Literal['gpu']]
    gpu_no: Optional[int]

    def __repr__(self):
        return (self.device_type, self.gpu_no).__repr__()


# TODO: not in use
# class ValidationConfig(TypedDict, total=False):
#     # Add validation options
#     ref: str  # id for this cross validation sequence
#     type: str  # eg KFold
#     n_splits: int  # number of splits to use in the data for KFold validation
#     split: int  # log index of split (out of n_splits)
#     n_repeats: int  # number of random initialisations for each fold.
#     iteration: int  # log index of current iteration (out of n_repeats)


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