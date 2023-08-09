import json
import os
import pickle
import wandb
import random
import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
from shared.config import BaseConfig, Config, DataSplitConfig, LoggingConfig
from shared.compilation.text_functor import TextFunctor
from shared.compilation.frame import DecomposeFrame


def get_filename(
        conf: BaseConfig,
        name: str = "train",
        model: bool = False,
        index: int = 0,
        artifact: bool = False,
        processed: Optional[str] = None,
):
    """
    Generate config-aware file names for models and processed datasets
    @param conf: {
        data.task,
        data.file_prefix,
        compilation : CompilationConfig  - for shared functor details,
        trainer.quantum : QuantumConfig  - for quantum model details,
        trainer.neural : NeuralConfig  - for neural model details,
    }
    @param name   dataset type (eg "train", "test")
    @param model  whether we are saving a model
    @param index  epoch index for the model
    @param artifact  generate a folder name
    @param processed  version info to append for processed datasets
    """
    quantum_extension = []
    # todo: add neural extension
    if "trainer" in conf and "quantum" in conf["trainer"]:
        qconf = conf["trainer"]["quantum"]
        quantum_extension = ["-".join(
            [
                qconf["ansatz"].value,
                str(qconf['layers']),
                str(qconf['wire_dimension']),
            ]
            + (
                ['mixed' if qconf['mixed'] else 'pure']
                if qconf['mixed'] is not None
                else []
            )
            + (
                ['dA' if qconf['discard_ancillas'] else 'psA']
                if qconf['discard_ancillas'] is not None
                else []
            )
        )]

    compilation_extension = []
    if "compilation" in conf:
        cconf = conf["compilation"]
        compilation_extension = [] + (
            [f"TF-{cconf['text_functor'].value}"] if cconf['text_functor'] is not TextFunctor.Id else []
        ) + (
            [f"HO-{cconf['frame_decomp'].value}"] if cconf['frame_decomp'] is not DecomposeFrame.NoDecomp else []
        ) + (
            [f"HOQ-{cconf['question_frame_decomp'].value}"] if cconf['question_frame_decomp'] is not DecomposeFrame.NoDecomp else []
        )

    config_extension = "_".join(
        ["task" + str(conf["data"]["task"])]
        + compilation_extension
        + quantum_extension
    )
    prefix = conf["data"]['file_prefix']
    dataset_name = f"{prefix}_dataset_{config_extension}_{name}.pkl"
    processed_dataset_name = f"{prefix}_dataset_{config_extension}_processed-{processed}_{name}.pkl"
    model_name = f"{prefix}_model_{config_extension}_{index}.pkl"
    artifact_name = f"{prefix}_{config_extension}"
    processed_artifact_name = f"{prefix}_{config_extension}_processed"

    if model:
        return model_name
    if artifact:
        return dataset_name, artifact_name
    if processed is not None:
        return processed_dataset_name, processed_artifact_name
    return dataset_name


def setup_env(config):
    # this should be relative path to \Neural-DisCoCirc
    PICKLED_DATASET_PATH = os.path.abspath(
        os.path.dirname(os.path.abspath(__name__)) + "/data/pickled_dataset"
    ) + "/"  # make sure it's a proper directory
    print("dataset path:", PICKLED_DATASET_PATH)
    config["env"]["dataset_path"] = PICKLED_DATASET_PATH

    # Set up pytorch config # todo: is this obsolete?
    if config["env"]["device"].device_type == "gpu":
        torch.set_default_tensor_type(torch.cuda.DoubleTensor)
    else:
        torch.set_default_tensor_type(torch.DoubleTensor)


def get_stratified_indices(config: DataSplitConfig, n: int = 1000, dataset_name: str = "train") -> List[int]:
    """
    Compute the stratified indices from the config.
    @param config: {
        strata_indices_train - Specify custom indices to use,
        strata_indices_test - Specify custom indices to use,
        strata - Number of different strata, for computing default,
    }
    @param n: size of the dataset, if computing default strata
    """
    conf_indices = config["strata_indices_" + dataset_name]
    if conf_indices is not None:
        if isinstance(conf_indices, list):
            return config["strata_indices_" + dataset_name]
        elif isinstance(conf_indices, str):
            # attempt to load this as an absolute file path
            try:
                with open(conf_indices, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"couldn't preload {dataset_name} indices:", e)
        else:
            pass
    # Otherwise, assume we are computing increasing context depth
    print("Assuming modular strata")
    return [i % config["strata"] for i in range(n)]


def save_indices(indices: List[int], name: str, config: LoggingConfig):
    """
    Save indices used for model training
    @param indices - list of indices to save
    @param name - name of dataset indices are for
    @param config.use_wandb - whether to save to wandb run
    @param config.local_path - path to save to if not saving to wandb
    """
    path = config["local_path"] if not config["use_wandb"] else wandb.run.dir
    os.makedirs(path, exist_ok=True)
    filename = os.path.join(path, f"{name}_indices.json")
    with open(filename, "w") as f:
        json.dump(indices, f)

    if config["use_wandb"]:
        wandb.save(filename)


def get_indices(name: str, config: LoggingConfig) -> List[int]:
    """
    Load previously saved indices
    @param name - name of dataset indices are for
    @param config.use_wandb - whether to get from wandb run
    @param config.wandb_project - if using wandb, define the run to query from
    @param config.local_path - path to query if not using wandb
    """
    path = config["local_path"]
    filename = f"{name}_indices.json"
    try:
        if config["use_wandb"]:
            wandb.restore(filename, run_path=config["wandb_project"].run_path(), root=path)
        else:
            with open(f"{path}/{filename}", "r") as f:
                print("Getting local indices")
    except Exception as e:
        print(e)
        print("Assume default indices")
        path = os.path.dirname(__file__)

    with open(f"{path}/{filename}", "r") as f:
        indices = json.load(f)
    return indices


def get_base_config(config: Config) -> BaseConfig:
    """
    Get a config that we can pass to get_filename to get the base dataset, before any compilations are applied.
    """
    return BaseConfig(
        data=config["data"],
        logging=config["logging"],
        env=config["env"],
    )


def get_data(config: BaseConfig, include: List[str] = ["train"], truncate: Optional[int] = None) -> Dict[str, list]:
    """
    Check the datasets for the given config are present.
    @param config
    @param exclude: specify datasets to skip over if not needed
    @param truncate: specify max examples to load. Default loads all datapoints.
    """
    _, artifact_name = get_filename(config, artifact=True)
    try:
        for key in include:
            filename = get_filename(config, name=key)
            # Check all the required files exist
            if config["logging"]["use_wandb"]:
                api = wandb.Api()
                artifact = api.artifact(f'{config["logging"]["wandb_project"].project_path()}/{artifact_name}:latest')
                artifact.get_path(filename)
            else:
                # Check if we have the local data
                if not os.path.isfile(config["env"]["dataset_path"] + filename):
                    raise Exception("No local file!", config["env"]["dataset_path"] + filename)

    except Exception as e:
        print(e)
        print("No pickled data for the given config!")
        raise Exception("Please convert from initial diagrams.")

    # Load the data
    return load_data(config, include=include, truncate=truncate)


def load_custom_dataset(file_path: str, truncate: Optional[int] = None) -> list:
    """Load dataset from a specified location."""
    print("Loading pickled dataset from:", file_path)
    with open(file_path, "rb") as f:
        # dataset is a tuple (context_circuit,(question_word_index, answer_word_index))
        dataset = []
        datum = pickle.load(f)
        # todo: check this case distinction works for neural
        if not isinstance(datum, list):
            print("dataset is split")
            dataset.append(datum)
            # Need to unpickle each entry
            i = 0
            try:
                while truncate is None or i < truncate:
                    if i % 10 == 0:
                        print(i, end="\r")
                    i += 1
                    dataset.append(pickle.load(f))
            except EOFError as e:
                print("Stopped loading data at index", i, e)
        else:
            print("entire dataset loaded")
            dataset = datum[:truncate] if truncate is not None else datum
    print("Loaded", len(dataset), "data points.")
    return dataset


def load_data(
        config: BaseConfig,
        run: Optional[wandb.run] = None,
        use_only: bool = False,
        include: List[str] = ["train"],
        truncate: Optional[int] = None,
) -> Optional[Dict[str, list]]:
    """
    Load quantum circuits to be evaluated.
    """
    datasets = {
        k: []
        for k in include
    }
    _, artifact_name = get_filename(config, artifact=True)

    if config["logging"]["use_wandb"]:
        api = wandb.Api()
        artifact = api.artifact(f'{config["logging"]["wandb_project"].project_path()}/{artifact_name}:latest')
        if run is not None:
            run.use_artifact(artifact)

    if not use_only:
        for key in include:
            filename = get_filename(config, name=key)
            if config["logging"]["use_wandb"]:
                artifact.get_path(filename).download(config["env"]["dataset_path"])

            datasets[key] = load_custom_dataset(config["env"]["dataset_path"] + filename, truncate)

        return datasets


def restrict_data(
        config: DataSplitConfig,
        train_indices: List[int],
        validation_indices: List[int],
        dataset_train_strata: List[int]
) -> Tuple[List[int], List[int]]:
    if config["restrict_train_data"] is not None:
        train_indices = [
            j for j in train_indices
            if dataset_train_strata[j] in config["restrict_train_data"]
        ]
        if config["restrict_valid_data"]:
            validation_indices = [
                j for j in validation_indices
                if dataset_train_strata[j] in config["restrict_train_data"]
            ]
    return train_indices, validation_indices


def init_seeds(seed: Optional[int] = None):
    if seed is not None:
        # todo add keras seed
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        print("Init seed", seed)
    else:
        torch.seed()
        np.random.seed()
        random.seed()
        print(f"Torch seed: {torch.initial_seed()}\n")
