import os
import torch
import json
import wandb
from pprint import pprint
from shared.utils import (
    get_filename,
    setup_env,
    get_data,
    load_data,
    load_custom_dataset,
    get_stratified_indices,
    get_indices,
    get_base_config,
)
from shared.trainer import SavedParams, Trainer
from shared.config import Config
api = wandb.Api()


def evaluate_model(
    config: Config = None,
    init_model: SavedParams = None,
    use_wandb=True,
    datasets=["test"],
    dataset_size=1000,
    custom_dataset=None,
    override_config={},
    resume=False,
):
    """
    Evaluate a specified model from a given run or file.
    @param config: run/evaluation config, if not using wandb.
    @param config.wandb_project: required if resuming a previous evaluation run.
    @param init_model: required if config not supplied. Specify where to obtain the model to be evaluated.
    @param use_wandb: Whether to use wandb.
    @param datasets: datasets to evaluate on.
        Allowed values:
            - test
            - valid
            - train
            - custom
            - {test, train, custom}_stratified  (to compute accuracies for each stratum)
            - {test, train, custom}.stratified.{int} (to target a particular stratum)
    @param custom_dataset: supply a custom dataset to evaluate the model on.
        Must be a filepath from which to load the data.
    @param dataset_size: number of data points in dataset. Used when calculating default strata.
    @param override_config: allow config overrides. Up to user to pick things that make sense.
    @param resume: if true, resume `config["wandb_project"]`
    """
    print("-- Begin Evaluation --")
    if config is None and not resume and init_model is None:
        raise ValueError(
            "You must provide a SavedParams initialiser for the model, "
            "either as `config['init_params']` or via `init_model`, if you intend the "
            "config to be retrieved from the model's wandb run."
        )
    # If using wandb get the config from there if its missing:
    if use_wandb and (config is None or resume):
        print("Loading config from wandb")
        run_path = init_model.wandb_path.run_path() if not resume else config["logging"]["wandb_project"].run_path()
        run = api.run(run_path)
        config = Config(**run.config)
        if init_model is not None:
            config["trainer"]["init_params"] = init_model
    print("Config:")
    pprint(config)
    print()

    print("Overriding config values:", override_config)
    config_prev = {**config, "custom_dataset_path": custom_dataset}  # record which custom dataset is used.
    config = {**config, **override_config}

    # Prepare datasets
    eval_dataset_names = set()
    eval_dataset_indices = []
    for dataset_type in datasets:
        dataset_name = "test" if "test" in dataset_type else (
            "custom" if "custom" in dataset_type else "train"
        )
        eval_dataset_names.add(dataset_name)

        # Generate the stratified dataset names
        if "_stratified" in dataset_type:
            for i in range(config["data_split"]["strata"]):
                base = dataset_type.split("_")[0]
                datasets.append(f"{base}.stratified.{i}")
            continue

        # dataset = dataset_test if dataset_name == "test" else dataset_train
        if dataset_type == "train":
            indices = get_indices("train", config)
            is_seen = True
        elif dataset_type in ["validation", "valid"]:
            indices = get_indices("valid", config)
            is_seen = True
        elif ".stratified." in dataset_type:
            level = int(dataset_type.split(".")[2])
            stratified_indices = get_stratified_indices(config, n=dataset_size, dataset_name=dataset_name)
            indices = [
                j for j in range(dataset_size)
                if stratified_indices[j] == level
            ]
            is_seen = dataset_name == "train" and (
                    config["data_split"]["restrict_train_data"] is None
                    or level in config["data_split"]["restrict_train_data"]
                    or not config["data_split"]["restrict_valid_data"]
            )
        else:
            raise Exception(f"Unidentified dataset type: {dataset_type}")
        eval_dataset_indices.append((dataset_name, dataset_type, indices, is_seen))

    # Compute dataset accuracies if not present.
    accuracy = {key: None for key in eval_dataset_names}
    if config["logging"]["use_wandb"] and resume:
        print("Getting accuracies from wandb:")
        # try downloading accuracies from wandb
        accuracy = {
            key: run.summary[key] if key in run.summary else None
            for key in eval_dataset_names
        }

    setup_env(config)

    if use_wandb:
        if resume:
            print("Resuming evaluation run")
            run = wandb.init(
                project=config["logging"]["wandb_project"].project,
                job_type="test_model",
                config={
                    **config_prev,
                    "config_overrides": override_config
                },
                resume="must",
                id=config["logging"]["wandb_project"].run
            )
        else:
            print("Initialise wandb run...")
            run = wandb.init(
                project=config["logging"]["wandb_project"].project,
                job_type="test_model",
                config={
                    **config_prev,
                    "config_overrides": override_config
                },
                tags=["model-evaluation"],
            )
            run.name = "evaluate model"
            # run.config["wandb_project"] = WandbProject(
            #     entity=config["wandb_project"].entity,
            #     project=config["wandb_project"].project,
            #     run=run.id
            # )   # todo: pretty sure wandb doesn't like this

    missing = sum([accuracy[key] is None for key in eval_dataset_names]) > 0
    if missing:
        print("Computing the dataset accuracies...")
        # Load datasets
        temp = get_data(
            get_base_config(config), include=list(eval_dataset_names.difference("custom"))
        )
        dataset_train, dataset_test = temp["train"], temp["test"]
        temp = None  # free memory?
        if custom_dataset is not None:
            # attempt to load from file path
            dataset_custom = load_custom_dataset(custom_dataset)
        else:
            dataset_custom = None

        datasets = {
            "train": dataset_train,
            "test": dataset_test,
            "custom": dataset_custom,
        }

        if config["logging"]["use_wandb"]:
            # Make sure to register the artifact with wandb
            load_data(config, run=run, use_only=True)

        print("Initialise trainer...")
        trainer: Trainer = config["trainer"]["trainer_class"](
            dataset_train,
            config,
        )

        # Compile the datasets
        for key in datasets.keys():
            if key == "custom":
                # Get a more specific name
                name = "custom__" + custom_dataset.split('/')[-1]
            else:
                name = key

            dataset = trainer.compile(
                datasets[key], config["compilation"],
                logging_config=config["logging"], data_config=config["data"], name=name
            )
            datasets[key] = dataset

        # Save the final model params
        filename = os.path.join(
            config["logging"]["local_path"] if not config["logging"]["use_wandb"] else wandb.run.dir,
            get_filename(config, model=True, index=0)
        )
        trainer.save(filename)
        if config["logging"]["use_wandb"]:
            wandb.save(filename)
        trainer.to(config["env"]["device"])

        # Evaluate model on entire dataset and log accuracies for each datapoint
        print("Evaluating model. Datasets:", eval_dataset_names)
        accuracy = {}
        for dataset_name in list(eval_dataset_names):
            if dataset_name not in accuracy or accuracy[dataset_name] is None:
                print("    - on", dataset_name, end="...")
                accuracy[dataset_name] = [
                    trainer.evaluate([datum], show_progress=False)
                    for datum in datasets[dataset_name]
                ]
                print("finished", len(accuracy[dataset_name]))
        if config["logging"]["use_wandb"]:
            wandb.log(accuracy)
        else:
            # Dump to a file
            with open(config["logging"]["local_path"] + "accuracy.json", "w") as f:
                json.dump(accuracy, f)

    # Compute the accuracies for the dataset segments specified
    acc_table = []
    for dataset_name, dataset_type, indices, is_seen in eval_dataset_indices:
        print(f"Calculating model accuracy on {dataset_type} dataset:")
        accs = [accuracy[dataset_name][i][0] for i in indices]
        skips = sum([accuracy[dataset_name][i][1] for i in indices])
        # Flag cases where there were no evaluated datapoints for the requested dataset
        if len(accs) == 0:
            print("No datapoints evaluated for dataset ", dataset_name)
            acc = None
        else:
            acc = sum(accs) / len(accs)
        acc_table.append([
            dataset_name,
            # Record the level if applicable, otherwise the full dataset identifier
            dataset_type.split(".")[2] if ".stratified." in dataset_type else dataset_type,
            acc,
            is_seen,
        ])
        print("skipped", skips, "accuracy", acc)
        if use_wandb:
            wandb.log({dataset_type: {"skipped": skips, "accuracy": acc}}, commit=False)

    if use_wandb:
        wandb_table = wandb.Table(columns=["dataset", "level", "accuracy", "seen"], data=acc_table)
        wandb.log({"acc_table":  wandb_table})
        run.finish()
    else:
        pprint(acc_table)

    print("FINISHED.")
