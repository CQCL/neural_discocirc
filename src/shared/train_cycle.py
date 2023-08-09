import os
import pickle
import wandb
from multiprocessing import Pool, Manager
from sklearn.model_selection import train_test_split, StratifiedKFold
from shared.utils import (
    get_filename,
    setup_env,
    get_data,
    load_data,
    get_stratified_indices,
    save_indices,
    get_indices,
    restrict_data,
    get_base_config,
    init_seeds,
)
from shared.trainer import SavedParams, Trainer
from shared.config import Config, ValidationConfig, WandbProject
api = wandb.Api()


def train(config: Config, trainer: Trainer, training_data, validation_data, start_epoch=0):
    """
    Training loop for a model, given pre-prepared data.
    Assume trainer implements the following methods:
        init_training(config, training_data) - function to initialise loss function and optimiser.
        fit(data, start_epoch, epoch_callback) - train the model
    """

    print("Training...")
    save_rate = config["logging"]["save_rate"]

    def epoch_callback(epoch, loss, gradients):
        if config["logging"]["use_wandb"]:
            gradient_histograms = {
                key: wandb.Histogram([
                    # Some nouns are not involved => None gradient.
                    (v if v is not None else 0)
                    for v in val
                ])
                for key, val in gradients.items()
            }
        else:
            gradient_histograms = None

        print()
        # Record validation accuracy every epoch
        valid_accuracy, valid_skipped = trainer.evaluate(validation_data)

        epoch_logs = {
            "loss": loss,
            "valid_acc": valid_accuracy,
            "valid_skipped": valid_skipped,
            "grads": gradient_histograms
        }
        print(f"Valid skipped {valid_skipped}   valid acc {valid_accuracy}")

        # Save the current trainer state
        state_dict = os.path.join(
            config["logging"]["local_path"] if not config["logging"]["use_wandb"] else wandb.run.dir,
            get_filename(config, model=True, index=epoch)
        )
        trainer.save(state_dict)
        if config["logging"]["use_wandb"]:
            wandb.save(state_dict)

        # Check performance on training dataset
        if epoch % save_rate == 0 or epoch == config["trainer"]["epochs"] - 1:
            train_accuracy, train_skipped = trainer.evaluate(training_data)
            epoch_logs = {**epoch_logs, **{
                "train_acc": train_accuracy,
                "train_skipped": train_skipped,
            }}
            print(f"Train skipped {train_skipped}   train acc {train_accuracy}")

        if config["logging"]["use_wandb"]:
            wandb.log(epoch_logs, step=epoch, commit=True)
        else:
            with open(config["logging"]["local_path"] + '/epoch_logs.pkl', 'ab') as f:
                pickle.dump([epoch, epoch_logs], f)

        print(f"End Epoch: {epoch}    epoch_loss: {loss}")

    trainer.fit(training_data, start_epoch=start_epoch, epoch_callback=epoch_callback)

    return trainer


def basic_run(
        config: Config,
        repeat=1,
        resume=False,
        resume_epoch=0,
        debug=None,
        n_processes=1
):
    """
    Basic training of a model for a given config.
    @param config
    @param resume: resume previous run. Specify run details in the config.
    @param resume_epoch: epoch from which to load the model if resuming.
    @param debug: load only debug data entries to speed things up.
    @param n_processes: for running mutliple iterations concurrently.
    """

    # Load datasets
    setup_env(config)
    dataset_train = get_data(get_base_config(config), include=["train"], truncate=debug)["train"]

    dataset_train_indices = list(range(len(dataset_train)))
    dataset_train_strata = get_stratified_indices(config["data_split"], n=len(dataset_train_indices))

    if resume:
        train_indices = get_indices("train", config["logging"])
        validation_indices = get_indices("valid", config["logging"])
    else:
        # Default train/test split
        train_indices, validation_indices = train_test_split(
            dataset_train_indices,
            test_size=0.1,
            random_state=config["data_split"]["data_split_seed"] if config["data_split"]["data_split_seed"] > -1 else None,
            stratify=dataset_train_strata,
        )
        # Apply dataset restrictions
        train_indices, validation_indices = restrict_data(
            config["data_split"], train_indices, validation_indices, dataset_train_strata
        )

    if n_processes == 1 and repeat == 1:
        # Don't use pools if there's only one thing to run.
        # Assume parent handles memory management.
        do_basic_run(
            config,
            train_indices,
            validation_indices,
            dataset_train,
            resume,
            resume_epoch,
        )
    else:
        with Pool(processes=n_processes, maxtasksperchild=1, initializer=init_seeds) as pool, Manager() as manager:
            shared_dataset_train = manager.list(dataset_train)
            for _ in range(repeat):
                pool.apply_async(
                    do_basic_run,
                    (
                        config,
                        train_indices,
                        validation_indices,
                        shared_dataset_train,
                        resume,
                        resume_epoch,
                    ),
                    callback=lambda: print("Done!"),
                    error_callback=lambda e: print("D'oh! ", e)
                )
            pool.close()
            pool.join()
    print("FINISHED")


def do_basic_run(
    config,
    train_indices,
    validation_indices,
    dataset_train,
    resume=False,
    resume_epoch=0,
):
    """
    Run training
    @param config
    @param train_indices
    @param validation_indices
    @param dataset_train: training dataset to use
    @param resume: whether to resume a previous run
    @param resume_epoch: epoch to resume previous run at
    """
    cv_config = config["cross_validation"]
    if cv_config is not None:
        print(f"\nFold {cv_config['split']}/{cv_config['n_splits'] - 1}, Iteration {cv_config['iteration']}/{cv_config['n_repeats']}\n")

    run = None
    if config["logging"]["use_wandb"]:
        print("Initialise wandb run...")
        run = wandb.init(
            project=config["wandb_project"].project,
            job_type="training_cycle",
            config=config,
            tags=["model-training"] if cv_config is None else ["cross-validation"],
            resume="allow",
            id=config["wandb_project"].run,
        )
        run.name = "train model"
    if resume:
        print("\nResuming run\n")

    if not resume:
        # Save the specific indices used for train/validation
        save_indices(train_indices, "train", config["logging"])
        save_indices(validation_indices, "valid", config["logging"])

    print("dataset size: train", len(train_indices), "     validation", len(validation_indices))

    # Make sure to register the artifact with wandb
    if config["logging"]["use_wandb"] and not resume:
        run.config["data_split"]["train_data_size"] = len(train_indices)
        run.config["data_split"]["valid_data_size"] = len(validation_indices)
        load_data(config, run=run, use_only=True)

    print("Initialise trainer...")
    # Supply appropriate init params if resuming
    if resume:
        saved_params = SavedParams(
            model_name=get_filename(config, model=True, index=resume_epoch),
            use_wandb=config["logging"]["use_wandb"],
            wandb_path=config["logging"]["wandb_project"],
            local_path=config["logging"]["local_path"],
        )
        trainer = config["trainer"]["trainer_class"].load(saved_params, config["trainer"], dataset_train)
    else:
        trainer = config["trainer"]["trainer_class"](
            dataset_train,
            config["trainer"],
        )

    # Process the datasets
    processed_dataset_train = trainer.compile_data(
        dataset_train, config["compilation"],
        data_name="train", logging_config=config["logging"], data_config=config["data"]
    )

    # restrict the datasets
    datasets = {
        "train": [processed_dataset_train[i] for i in train_indices],
        "valid": [processed_dataset_train[i] for i in validation_indices],
    }

    trainer.to(config["env"]["device"])
    if resume_epoch == 0:
        filename = os.path.join(
            config["logging"]["local_path"] if not config["logging"]["use_wandb"] else wandb.run.dir,
            get_filename(config, model=True, index=-1)
        )
        # trainer.save(filename)
        if config["logging"]["use_wandb"]:
            run.save(filename)

    # try:
    trainer = train(
        config,
        trainer,
        datasets["train"],
        datasets["valid"],
        start_epoch=resume_epoch + 1 if resume else 0
    )
    # except Exception as e:
    #     print(f"RUN {run.id if run else ''} FAILED:\n", e)

    if config["logging"]["use_wandb"]:
        run.finish()
    return trainer


def cross_validate(
        config: Config,
        resume=False,
        resume_split=0,
        resume_iteration=0,
        n_processes=1  # Run concurrently
):
    """
    Run cross validation for a given config.
    @param config
    @param resume: resume previous cross validation run.
        Note, this does not resume a cross validation iteration.
    @param resume_split: cv split to resume from
    @param resume_iteration: iteration to resume from within the specified split/fold.
    @param n_processes: for running mutliple iterations concurrently.
    """
    setup_env(config)
    cv_config = config["cross_validation"]

    # Load datasets
    dataset_train = get_data(get_base_config(config), include=["train"])["train"]
    dataset_train_indices = list(range(len(dataset_train)))
    dataset_train_strata = get_stratified_indices(config["data_split"], n=len(dataset_train_indices))

    kf = StratifiedKFold(
        n_splits=cv_config["n_splits"],
    ).split(dataset_train_indices, dataset_train_strata)

    # Run k-fold cross validation on the training data
    print("Start iterations")
    with Pool(processes=n_processes, maxtasksperchild=1, initializer=init_seeds) as pool, Manager() as manager:
        shared_dataset_train = manager.list(dataset_train)
        for i, (train_indices, validation_indices) in enumerate(kf):
            if resume and i < resume_split:
                continue

            # Get back the actual indices
            train_indices = [dataset_train_indices[j] for j in train_indices]
            validation_indices = [dataset_train_indices[j] for j in validation_indices]
            # Apply dataset restrictions
            train_indices, validation_indices = restrict_data(
                config["data_split"], train_indices, validation_indices, dataset_train_strata
            )

            cv_config["split"] = i
            for k in range(cv_config["n_repeats"]):
                if resume and resume_split == i and k < resume_iteration:
                    continue

                cv_config["iteration"] = k
                config["cross_validation"] = ValidationConfig(**cv_config)
                pool.apply_async(do_basic_run, (
                    {**config},
                    [t for t in train_indices],
                    [v for v in validation_indices],
                    shared_dataset_train,
                    False,  # resume
                    0,  # resume epoch
                ))

        pool.close()
        # wait for pool to finish.
        pool.join()
    print("FINISHED")
