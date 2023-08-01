import os.path
import torch
from pprint import pprint
from config.config import get_config

from config.neural_config import get_neural_config
from config.utils import Device, WandbProject
from models.add_logits_model import AddLogitsModel
from train_model import basic_run
from trainers.one_network_trainer import OneNetworkTrainer

# torch device
# device = Device(device_type="gpu" if torch.cuda.is_available() else "cpu")
device = Device(device_type="cpu", gpu_no=None)

config = get_config(
    neural=get_neural_config(),

    # Task and dataset
    task=1,
    # file_prefix="isin",
    # text_functor=TextFunctor.RemoveThe,
    # frame_decomp=DecomposeFrame.NoDiscards,
    # question_frame_decomp=DecomposeFrame.NameOnly,
    # override_dataset_train_path: str
    # override_dataset_test_path: str
    
    # strata=5,
    # strata_indices_train: Union[None, str, List[int]]
    # strata_indices_test: Union[None, str, List[int]]
    # strata_indices_custom: Union[None, str, List[int]]  # used when specifying custom eval datasets

    trainer_class=OneNetworkTrainer,
    model_class=AddLogitsModel,
    # init_params=CliffordParams(0, 4, 0.5),  # Default is uniform [0,2]

    epochs=10,
    learning_rate=0.001,
    batch_size=32,
    device=device,
    train_data_used=0,
    restrict_train_data=[0],
    restrict_valid_data=True,
    data_split_seed=0,

    save_rate=3,
    use_wandb=False,
    wandb_project=WandbProject(
        entity="domlee",
        project="discocirc",
        run=None
    ),
    
    # local_path=os.path.abspath('./temp/0007')
    # dataset_path: str  # path to saved datasets. Includes trailing /


    # cross_validate=ValidationConfig(
    #     type="KFold",
    #     n_splits=5,
    #     n_repeats=5,
    #     ref="CV_blabla",
    # )
)
pprint(config)


# Run stuff:
# cross_validate(config)
basic_run(
    config,
    # debug=None,
    # debug=50,  # truncate dataset
    # resume=False,
    # resume_epoch=0,
)