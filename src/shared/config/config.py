from typing import Optional, List, Union
from dataclasses import dataclass
from shared.config.quantum_config import QuantumConfig, get_quantum_config
from shared.compilation.frame import DecomposeFrame
from shared.compilation.text_functor import TextFunctor
from shared.trainer import Trainer, InitParams
from .base_classes import Model
from .utils import dictify, Device, ValidationConfig, WandbProject, EnvConfig, LoggingConfig


@dictify
@dataclass(kw_only=True)
class DataConfig:
    task: Union[str, int] = 1  # Todo: Allow this to be a list for multi-task training?
    file_prefix: str = ''  # Filename prefix for all dataset and model files generated.
    # load specified (pre-prepared) datasets directly, for when we don't want to use the config-generated file names.
    # Supplied to the model as-is.
    # todo: implement
    override_dataset_train_path: Optional[str] = None
    override_dataset_test_path: Optional[str] = None


@dictify
@dataclass(kw_only=True)
class CompilationConfig:
    text_functor: TextFunctor = TextFunctor.Id  # Functor to apply to the diagrams before processing
    frame_decomp: DecomposeFrame = DecomposeFrame.NoDecomp  # How higher order boxes are to be decomposed.
    question_frame_decomp: DecomposeFrame = DecomposeFrame.NoDecomp  # Decompose higher order boxes in the question


@dictify
@dataclass
class TrainerConfig:
    # Model
    trainer_class: Trainer = Trainer  # Trainer class to instantiate.
    model_class: Model = Model  # Model class to instantiate.
    init_params: InitParams = InitParams()  # Sampling strategy to use when initialising model parameters.
    # Training
    epochs: int = 10
    learning_rate: float = 0.0005
    batch_size: int = 1
    quantum: Optional[QuantumConfig] = None
    # neural: Optional[NeuralConfig] = None


@dictify
@dataclass(kw_only=True)
class DataSplitConfig:
    data_split_seed: int = -1  # seed to use to split between training/validation. Use -1 for unseeded.
    train_data_used: int = 0  # speed up training by using only part of the available data (shuffled each epoch). 0 means use all of it
    restrict_train_data: Optional[List[int]] = None  # Restrict training data used to specified strata; start at 0
    restrict_valid_data: Optional[bool] = False  # whether to restrict validation data like training.
    # Dataset characterisation:
    # Dataset classes are calculated as (index % strata) if indices aren't specified.
    strata: int = 1  # number of strata in the dataset
    # Define custom splitting of the dataset into classes. Must match the dataset length.
    # Can be either a file reference or an explicit list
    strata_indices_train: Union[None, str, List[int]] = None
    strata_indices_test: Union[None, str, List[int]] = None
    strata_indices_custom: Union[None, str, List[int]] = None  # used when specifying custom eval datasets


@dictify
@dataclass(kw_only=True)
class BaseConfig:
    data: DataConfig = DataConfig()
    logging: LoggingConfig = LoggingConfig()
    env: EnvConfig = EnvConfig()


@dataclass(kw_only=True)
class Config(BaseConfig):
    trainer: TrainerConfig = TrainerConfig()
    compilation: CompilationConfig = CompilationConfig()
    data_split: DataSplitConfig = DataSplitConfig()
    cross_validation: Optional[ValidationConfig] = None


def get_config(**kwargs):
    """Generate a config object with defaults."""
    default_config = Config(
        quantum=None,
        # neural=None,

        # Task and dataset
        task=1,
        file_prefix='isin',
        text_functor=TextFunctor.Id,
        frame_decomp=DecomposeFrame.NoDecomp,
        question_frame_decomp=DecomposeFrame.NoDecomp,

        strata=1,  # Group all data together
        strata_indices_train=None,
        strata_indices_test=None,
        strata_indices_custom=None,

        # Model
        trainer_class=Trainer.adam,
        model_class=Model.context,
        init_params=UniformParams(0, 2),

        # Training
        epochs=15,
        learning_rate=0.0005,
        batch_size=1,
        device=Device(device_type='cpu', gpu_no=None),
        train_data_used=0,
        restrict_train_data=None,
        restrict_valid_data=False,
        data_split_seed=-1,

        # Logging
        save_rate=5,
        use_wandb=False,
        wandb_project=None,
        local_path="../../quantum/config",  # relative local path if unspecified
        dataset_path="../../quantum/config",

        # Cross validation
        cross_validation=None
    )
    for kwarg, val in kwargs.items():
        if kwarg in default_config:
            # Ensure enum types are mapped correctly
            if kwarg == "frame_decomp" and not isinstance(val, DecomposeFrame):
                val = DecomposeFrame[str(val).split(".")[-1]]
            if kwarg == "question_frame_decomp" and not isinstance(val, DecomposeFrame):
                val = DecomposeFrame[str(val).split(".")[-1]]
            if kwarg == "device" and not isinstance(val, Device):
                if isinstance(val, str):
                    val = Device(val, None)
                else:
                    val = Device(*val)
            if kwarg == "init_params" and not isinstance(val, InitParams):
                val = get_init_params_from_str(val)
            if kwarg == "trainer_class" and not isinstance(val, Trainer):
                val = Trainer[str(val).split(".")[-1]]
            if kwarg == "model_class" and not isinstance(val, Model):
                val = Model[str(val).split(".")[-1]]
            if kwarg == "text_functor" and not isinstance(val, TextFunctor):
                val = TextFunctor[str(val).split(".")[-1]]
            if kwarg == "quantum":
                val = get_quantum_config(**val)
            if kwarg == "wandb_project" and not isinstance(val, WandbProject):
                val = WandbProject(*val)

            default_config[kwarg] = val
        else:
            # might be from an old config format
            default_config = port_forward(kwarg, val, default_config)

    return default_config


default_q_config = get_quantum_config()


# Backwards compatible config
def port_forward(kwarg, val, config):
    if kwarg in default_q_config:
        qconf = get_quantum_config(kwarg=val)
        if config["quantum"] is None:
            config["quantum"] = qconf
        else:
            config["quantum"][kwarg] = qconf[kwarg]
    return config
