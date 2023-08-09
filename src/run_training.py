import os.path
from pprint import pprint
from neural.models.add_logits_model import AddLogitsModel
from neural.trainers.one_network_trainer import OneNetworkTrainer

from shared import (
    DecomposeFrame,
    TextFunctor,
    Config,
    Device,
    WandbProject,
    TrainerConfig,
    DataConfig,
    DataSplitConfig,
    CompilationConfig,
    EnvConfig,
    LoggingConfig,
    basic_run,
    NeuralConfig,
)


# torch device:
# device = Device(device_type="gpu" if torch.cuda.is_available() else "cpu")
device = Device(device_type="cpu", gpu_no=None)

config = Config(
    trainer=TrainerConfig(
        trainer_class=OneNetworkTrainer,
        model_class=AddLogitsModel,
        init_params=None,
        learning_rate=0.001,
        batch_size=5,
        epochs=10,
        neural=NeuralConfig(),
    ),
    data=DataConfig(
        task=1,
        file_prefix="neural",
        override_dataset_train_path=None,
        override_dataset_test_path=None,
    ),
    compilation=CompilationConfig(
        text_functor=TextFunctor.RemoveThe,
        frame_decomp=DecomposeFrame.NoDiscards,
        question_frame_decomp=DecomposeFrame.NameOnly,
    ),
    data_split=DataSplitConfig(
        data_split_seed=0,
        train_data_used=0,
        restrict_train_data=[0],
        restrict_valid_data=True,
        strata=5,
        strata_indices_train=None,
        strata_indices_test=None,
        strata_indices_custom=None,
    ),
    env=EnvConfig(
        device=device,
    ),
    logging=LoggingConfig(
        save_rate=3,
        use_wandb=False,
        # use_wandb=False,
        wandb_project=WandbProject(
            entity="domlee",
            # project="compositionality",
            project="is_in_prob_dist_01",
            # run=None,
            run="3mo9c3uh"
        ),
        local_path=os.path.abspath('./temp/0007')
    ),
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
    debug=50,  # truncate dataset
    resume=False,
    resume_epoch=0,
)