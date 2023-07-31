import copy
import os
import shutil
from inspect import signature

from network.models.add_logits_model import AddLogitsModel
from network.models.add_scaled_logits_model import AddScaledLogitsModel
from network.models.is_in_max_wire_model import IsInMaxWireModel

from network.models.is_in_relation import IsInRelationModel
from network.models.lstm_model import LSTMModel
from network.models.textspace_model import TextspaceModel
from network.models.weighted_sum_of_wires_one_network import \
    WeightedSumOfWiresModel
from network.trainers.individual_networks_trainer import \
    IndividualNetworksTrainer
from network.trainers.one_network_trainer import OneNetworkTrainer

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
output_config = {
    "log_wandb": False,
    "wandb_project": "discocirc",
    "tb_callback": False,
    "save_model": True,
    "run_test_dataset": False,
}

training_config = {
    "batch_size": 32,
    "dataset_size": 10,  # -1 for entire dataset
    "epochs": 5,
    "learning_rate": 0.001,
    "model": AddLogitsModel,
    "task": 1,
    "trainer": OneNetworkTrainer,
    # "trainer": IndividualNetworksTrainer,
    "dataset_split": ("random", 1), # (split_type, random state)
    # "dataset_split": ("depth", [1]) # (split_type, depths of training set)
}

model_configs = {
    "wire_dimension": 10,
    "hidden_layers": [10, 10],
    "is_in_hidden_layers": [10, 10],
    "relevance_hidden_layers": [10, 10],
    "softmax_relevancies": False,
    "softmax_logits": False,
    "expansion_hidden_layers": [20, 50],
    "contraction_hidden_layers": [50, 20],
    "latent_dimension": 100,
    "textspace_dimension": 20,
    "qna_hidden_layers": [10, 10],
    "lstm_dimension": 10,
}


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


def train(base_path, save_path, vocab_path, data_path):
    model_class = training_config['model']
    print('Create model_config...')
    model_config = {}
    for val in signature(model_class.__init__).parameters:
        if val not in model_configs.keys():
            continue
        model_config[val] = model_configs[val]

    training_config.update(model_config)
    training_config['hidden_layers'] = model_configs['hidden_layers']
    if output_config["log_wandb"]:
        print("Initialise wandb...")
        wandb.init(project=output_config["wandb_project"], entity="domlee",
                   config=training_config)

    train_dataset_name = "task{:02d}_train.p".format(training_config["task"])

    print('Training: {} with trainer {} on data {}'
          .format(model_class.__name__,
                  training_config['trainer'].__name__,
                  train_dataset_name))

    vocab_file = base_path + vocab_path + "task{:02d}_train.p".format(training_config["task"])
    print('Loading vocabulary: {}'.format(vocab_file))
    with open(vocab_file, 'rb') as file:
        lexicon = pickle.load(file)

    dataset_file = base_path + data_path + train_dataset_name
    print('Loading pickled dataset: {}'.format(dataset_file))
    with open(dataset_file, "rb") as f:
        # dataset is a tuple (context_circuit,(question_word_index, answer_word_index))
        dataset = pickle.load(f)
        if training_config['dataset_size'] != -1:
            dataset = dataset[:training_config['dataset_size']]

    print('Splitting dataset...')
    if training_config['dataset_split'][0] == 'random':
        train_dataset, validation_dataset = train_test_split(dataset,
                                                         test_size=0.1,
                                                         random_state=training_config['dataset_split'][1])
    elif training_config['dataset_split'][0] == 'depth':
        train_dataset, validation_dataset = train_test_depth_split(dataset,
                                                            training_depths=training_config['dataset_split'][1])
    print('Initializing trainer...')
    discocirc_trainer = training_config['trainer'](lexicon=lexicon,
                            model_class=model_class,
                            hidden_layers=training_config['hidden_layers'],
                            question_length = len(dataset[0]['question']),
                            **model_config
    )

    discocirc_trainer.model_class.build([])
    discocirc_trainer.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=training_config["learning_rate"]),
        run_eagerly=True
    )

    datetime_string = datetime.now().strftime("%B_%d_%H_%M")
    validation_callback = ValidationAccuracy(discocirc_trainer.get_accuracy,
                    interval=1, log_wandb=output_config["log_wandb"])

    print('Training...')

    callbacks = [validation_callback]

    if output_config['tb_callback']:
        tb_callback = keras.callbacks.TensorBoard(
            log_dir='logs/{}'.format(datetime_string),
            histogram_freq=0,
            write_graph=True,
            write_images=True,
            update_freq='batch',
        )
        callbacks.append(tb_callback)

    if output_config['save_model']:
        save_base_path = base_path + "/checkpoints/"
        checkpoint_callback = ModelCheckpointWithoutSaveTraces(
            filepath='{}/{}'.format(save_base_path, datetime_string),
            save_freq=20 * training_config["batch_size"]
        )
        callbacks.append(checkpoint_callback)

    if output_config["log_wandb"]:
        callbacks.append(WandbCallback())

    discocirc_trainer.fit(
        train_dataset,
        validation_dataset,
        epochs=training_config['epochs'],
        batch_size=training_config['batch_size'],
        callbacks=callbacks
    )

    accuracy = discocirc_trainer.get_accuracy(discocirc_trainer.dataset)
    print("The accuracy on the train set is", accuracy)

    if output_config["log_wandb"]:
        wandb.log({"train_accuracy": accuracy})

    if output_config["run_test_dataset"]:
        print("Getting the test accuracy")
        test_dataset_name = "task{:02d}_test.p".format(training_config["task"])
        with open(base_path + data_path + test_dataset_name, 'rb') as f:
            test_dataset = pickle.load(f)

        test_accuracy = discocirc_trainer.get_accuracy(
            discocirc_trainer.compile_dataset(test_dataset))
        print("The accuracy on the test set is", test_accuracy)

        if output_config["log_wandb"]:
            wandb.log({"test_accuracy": test_accuracy})

    if output_config['save_model']:
        save_base_path = base_path + save_path + model_class.__name__
        Path(save_base_path).mkdir(parents=True, exist_ok=True)
        name = save_base_path + "/" + model_class.__name__ + "_" \
               + datetime.utcnow().strftime("%h_%d_%H_%M")

        discocirc_trainer.save(name, save_traces=False)

        shutil.make_archive(name, 'zip', name)

        if output_config["log_wandb"]:
            wandb.save(name + '.zip')


if __name__ == "__main__":
    train(base_path,
          "/saved_models/",
          '/data/task_vocab_dicts/',
          "/data/pickled_dataset/")
