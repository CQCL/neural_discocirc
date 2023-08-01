from typing import TypedDict, Optional, List, Literal, Union, NamedTuple
from models.add_logits_model import AddLogitsModel
from models.model_base_class import ModelBaseClass
from trainers.one_network_trainer import OneNetworkTrainer

from trainers.trainer_base_class import TrainerBaseClass

from .neural_config import NeuralConfig, get_neural_config
from .utils import Device, WandbProject


class Config(TypedDict):
    # Quantum
    # quantum: Optional[QuantumConfig]
    # Neural
    neural: Optional[NeuralConfig]  # Defined in neural part.

    # Task and dataset
    task: int  # shared. Allow this to be a list for multi-task training? (Would require the model to be applicable to both)
    file_prefix: str  # filename prefix for all dataset and model files this run will create.
    # text_functor: TextFunctor  # not currently used by neural, but can be shared. Functor to apply to the diagrams before processing
    # frame_decomp: DecomposeFrame  # how higher order boxes are to be decomposed. shared
    # question_frame_decomp: DecomposeFrame  # shared. decompose higher order boxes in the question
    # load specified (pre-prepared) datasets directly. Supplied to the model as-is.
    override_dataset_train_path: str
    override_dataset_test_path: str

    # Dataset characterisation
    # number of questions for the same (growing) context. shared
    # Dataset classes are calculated as (index % strata) if indices aren't specified.
    strata: int
    # Define custom splitting of the dataset into classes. Must match the dataset length.
    strata_indices_train: Union[None, str, List[int]]
    strata_indices_test: Union[None, str, List[int]]
    strata_indices_custom: Union[None, str, List[int]]  # used when specifying custom eval datasets

    # Model
    trainer_class: TrainerBaseClass  # Trainer class to instantiate. shared.
    model_class: ModelBaseClass  # Model class to instantiate. shared.
    # init_params: InitParams  # Sampling strategy to use when initialising model parameters. shared

    # Training
    epochs: int  # shared
    learning_rate: float  # shared
    batch_size: int  # shared
    device: Device  # device to target for training. shared, more generic type to accomodate other libraries.
    train_data_used: int  # speed up training by using only part of the available data (shuffled each epoch). shared, use for debug
    restrict_train_data: Optional[List[int]]  # Restrict training data used to specified strata; start at 0. shared
    restrict_valid_data: Optional[bool]  # whether to restrict validation data like training. Default is falsy. shared
    data_split_seed: int  # seed to use to split between training/validation. Use -1 for unseeded. shared.

    # logging
    save_rate: int  # how often to checkpoint the model. shared
    use_wandb: bool  # whether to log stuff to wandb. shared.
    wandb_project: Optional[WandbProject]  # path to wandb project. shared.
    local_path: Optional[str]  # local path to save files to if not using wandb
    dataset_path: str  # path to saved datasets. Includes trailing /

    # Cross validation
    # cross_validation: Optional[ValidationConfig]


def get_config(**kwargs):
    """Generate a config object with defaults."""
    default_config = Config(
        quantum=None,
        neural=None,

        # Task and dataset
        task=1,
        file_prefix='isin',
        # text_functor=TextFunctor.Id,
        # frame_decomp=DecomposeFrame.NoDecomp,
        # question_frame_decomp=DecomposeFrame.NoDecomp,

        strata=1,  # Group all data together
        strata_indices_train=None,
        strata_indices_test=None,
        strata_indices_custom=None,

        # Model
        trainer_class=OneNetworkTrainer,
        model_class=AddLogitsModel,
        # init_params=UniformParams(0, 2),

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
        local_path=".",  # relative local path if unspecified
        dataset_path=".",

        # Cross validation
        cross_validation=None
    )
    
    for kwarg, val in kwargs.items():
        if kwarg in default_config:
            # # Ensure enum types are mapped correctly
            # if kwarg == "frame_decomp" and not isinstance(val, DecomposeFrame):
            #     val = DecomposeFrame[str(val).split(".")[-1]]
            # if kwarg == "question_frame_decomp" and not isinstance(val, DecomposeFrame):
            #     val = DecomposeFrame[str(val).split(".")[-1]]
            # if kwarg == "device" and not isinstance(val, Device):
            #     if isinstance(val, str):
            #         val = Device(val, None)
            #     else:
            #         val = Device(*val)
            # if kwarg == "init_params" and not isinstance(val, InitParams):
            #     val = get_init_params_from_str(val)
            # if kwarg == "trainer_class" and not isinstance(val, Trainer):
            #     val = Trainer[str(val).split(".")[-1]]
            # if kwarg == "model_class" and not isinstance(val, Model):
            #     val = Model[str(val).split(".")[-1]]
            # if kwarg == "text_functor" and not isinstance(val, TextFunctor):
            #     val = TextFunctor[str(val).split(".")[-1]]
            # if kwarg == "quantum":
            #     val = get_quantum_config(**val)
            if kwarg == "wandb_project" and not isinstance(val, WandbProject):
                val = WandbProject(*val)

            default_config[kwarg] = val
        # else:
        #     # might be from an old config format
        #     default_config = port_forward(kwarg, val, default_config)

    return default_config