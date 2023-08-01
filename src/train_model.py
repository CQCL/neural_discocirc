import os
import shutil
from inspect import signature
from config.config import Config

from models.add_logits_model import AddLogitsModel
from models.add_scaled_logits_model import AddScaledLogitsModel
from models.is_in_max_wire_model import IsInMaxWireModel

from models.is_in_relation import IsInRelationModel
from models.lstm_model import LSTMModel
from models.textspace_model import TextspaceModel
from models.weighted_sum_of_wires_one_network import \
    WeightedSumOfWiresModel
from trainers.individual_networks_trainer import \
    IndividualNetworksTrainer
from trainers.one_network_trainer import OneNetworkTrainer

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import pickle
from datetime import datetime
from pathlib import Path

from tensorflow import keras
import wandb
from wandb.integration.keras import WandbCallback

from utils.callbacks import ValidationAccuracy, \
    ModelCheckpointWithoutSaveTraces
from sklearn.model_selection import train_test_split

# this should the path to \Neural-DisCoCirc
# base_path = os.path.abspath('..')
base_path = os.path.abspath('.')
save_path = "/saved_models/"
vocab_path = '/data/task_vocab_dicts/'
data_path = "/data/pickled_dataset/"


def train_test_depth_split(dataset, training_depths):
    split_datasets = []
    previous_length = 0
    counter = 0
    for q in dataset:
        if len(q['context_circ']) < previous_length:
            counter = 0
        previous_length = len(q['context_circ'])
        if len(split_datasets) <= counter:
            split_datasets.append([])
        split_datasets[counter].append(q)
        counter += 1

    training_dataset = []
    validation_dataset = []
    for i, set in enumerate(split_datasets):
        if i in training_depths:
            training_dataset += split_datasets[i]
        else:
            validation_dataset += split_datasets[i]

    return training_dataset, validation_dataset


def basic_run(config: Config):
    model_class = config['model_class']
    print('Create model_config...')
    model_config = {}
    for val in signature(model_class.__init__).parameters:
        if val not in config['neural'].keys():
            continue
        model_config[val] = config['neural'][val]

    # training_config.update(model_config)
    # training_config['hidden_layers'] = model_configs['hidden_layers']
    
    if config["use_wandb"]:
        print("Initialise wandb...")
        wandb.init(project=config["wandb_project"].project, entity=config["wandb_project"].entity,
                   config=config)

    train_dataset_name = "task{:02d}_train.p".format(config["task"])

    print('Training: {} with trainer {} on data {}'
          .format(model_class.__name__,
                  config['trainer_class'].__name__,
                  train_dataset_name))

    vocab_file = base_path + vocab_path + "task{:02d}_train.p".format(config["task"])
    print('Loading vocabulary: {}'.format(vocab_file))
    with open(vocab_file, 'rb') as file:
        lexicon = pickle.load(file)

    dataset_file = base_path + data_path + train_dataset_name
    print('Loading pickled dataset: {}'.format(dataset_file))
    with open(dataset_file, "rb") as f:
        # dataset is a tuple (context_circuit,(question_word_index, answer_word_index))
        dataset = pickle.load(f)
        # TODO
        # if training_config['dataset_size'] != -1:
        dataset = dataset[:10]

    print('Splitting dataset...')
    # TODO
    train_dataset, validation_dataset = train_test_split(dataset,
                                                         test_size=0.1,
                                                         random_state=1)

    print('Initializing trainer...')
    discocirc_trainer = config['trainer_class'](lexicon=lexicon,
                            model_class=model_class,
                            hidden_layers=config['neural']['hidden_layers'],
                            question_length = len(dataset[0]['question']),
                            **model_config
    )

    discocirc_trainer.model.build([])
    discocirc_trainer.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=config["learning_rate"]),
        run_eagerly=True
    )

    datetime_string = datetime.now().strftime("%B_%d_%H_%M")
    validation_callback = ValidationAccuracy(discocirc_trainer.get_accuracy,
                    interval=1, log_wandb=config["use_wandb"])

    print('Training...')
    callbacks = [validation_callback]

    if config["save_rate"] >= 0:
        save_base_path = base_path + "/checkpoints/"
        checkpoint_callback = ModelCheckpointWithoutSaveTraces(
            filepath='{}/{}'.format(save_base_path, datetime_string),
            save_freq=config["save_rate"] * config["batch_size"]
        )
        callbacks.append(checkpoint_callback)

    if config["use_wandb"]:
        callbacks.append(WandbCallback())

    discocirc_trainer.fit(
        train_dataset,
        validation_dataset,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        callbacks=callbacks
    )

    accuracy = discocirc_trainer.get_accuracy(discocirc_trainer.dataset)
    print("The accuracy on the train set is", accuracy)

    if config["use_wandb"]:
        wandb.log({"train_accuracy": accuracy})

    # if output_config["run_test_dataset"]:
    #     print("Getting the test accuracy")
    #     test_dataset_name = "task{:02d}_test.p".format(config["task"])
    #     with open(base_path + data_path + test_dataset_name, 'rb') as f:
    #         test_dataset = pickle.load(f)

    #     test_accuracy = discocirc_trainer.get_accuracy(
    #         discocirc_trainer.compile_dataset(test_dataset))
    #     print("The accuracy on the test set is", test_accuracy)

    #     if output_config["log_wandb"]:
    #         wandb.log({"test_accuracy": test_accuracy})

    if config['save_rate'] >= 0:
        save_base_path = base_path + save_path + model_class.__name__
        Path(save_base_path).mkdir(parents=True, exist_ok=True)
        name = save_base_path + "/" + model_class.__name__ + "_" \
               + datetime.utcnow().strftime("%h_%d_%H_%M")

        discocirc_trainer.save(name, save_traces=False)

        shutil.make_archive(name, 'zip', name)

        if config["use_wandb"]:
            wandb.save(name + '.zip')