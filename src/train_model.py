import copy
import os
import shutil
from inspect import signature

from network.models.add_logits_model import AddLogitsModel
from network.models.add_scaled_logits_model import AddScaledLogitsModel
from network.models.is_in_model import IsInModel
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
base_path = os.path.abspath('..')
# base_path = os.path.abspath('.')
output_config = {
    "save_model": False,
    "print_weights": False,
    "tb_callback": False,
}

training_config = {
    "batch_size": 32,
    "dataset_size": 20,  # -1 for entire dataset
    "dataset": "task01_train_dataset.pkl",
    "epochs": 20,
    "learning_rate": 0.01,
    "log_wandb": False,
    "model": AddScaledLogitsModel,
    # "trainer": OneNetworkTrainer,
    "trainer": IndividualNetworksTrainer,
    "lexicon": "en_qa1.p",
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


def print_weights(pre_training, post_training):
    print(len(pre_training), len(post_training))
    for weight in pre_training:
        found = False
        for w in post_training:
            if w.name == weight.name:
                print(w.name)
                print(weight - w)
                found = True
                break
        if not found:
            print("Weight not found: {}".format(weight.name))


def train(base_path, save_path, vocab_path,
          data_path):
    model_class = training_config['model']
    print('Create model_config...')
    model_config = {}
    for val in signature(model_class.__init__).parameters:
        if val not in model_configs.keys():
            continue
        model_config[val] = model_configs[val]

    training_config.update(model_config)
    if training_config["log_wandb"]:
        print("Initialise wandb...")
        wandb.init(project="discocirc", entity="domlee",
                   config=training_config)

    print('Training: {} with trainer {} on data {}'
          .format(model_class.__name__,
                  training_config['trainer'].__name__,
                  training_config["dataset"]))

    print('Loading vocabulary...')
    with open(base_path + vocab_path + training_config["lexicon"],
              'rb') as file:
        lexicon = pickle.load(file)

    print('Initializing trainer...')
    discocirc_trainer = training_config['trainer'](lexicon=lexicon,
                            model_class=model_class,
                            hidden_layers=model_configs['hidden_layers'],
                            **model_config
    )

    print('Loading pickled dataset...')
    with open(base_path + data_path + training_config['dataset'],
              "rb") as f:
        # dataset is a tuple (context_circuit,(question_word_index, answer_word_index))
        dataset = pickle.load(f)
        if training_config['dataset_size'] != -1:
            dataset = dataset[:training_config['dataset_size']]

    train_dataset, validation_dataset = train_test_split(dataset,
                                                         test_size=0.1,
                                                         random_state=1)

    discocirc_trainer.model_class.build([])
    discocirc_trainer.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=training_config["learning_rate"]),
        run_eagerly=True
    )

    datetime_string = datetime.now().strftime("%B_%d_%H_%M")
    validation_callback = ValidationAccuracy(discocirc_trainer.get_accuracy,
                    interval=1, log_wandb=training_config["log_wandb"])

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

    if training_config["log_wandb"]:
        callbacks.append(WandbCallback())

    model_weights = copy.deepcopy((discocirc_trainer.model_class.weights))
    trainer_weights = copy.deepcopy((discocirc_trainer.weights))

    discocirc_trainer.fit(
        train_dataset,
        validation_dataset,
        epochs=training_config['epochs'],
        batch_size=training_config['batch_size'],
        callbacks=callbacks
    )

    if output_config['print_weights']:
        print("----- Trainer weights: ------")
        print_weights(trainer_weights, discocirc_trainer.weights)

        print("----- Model weights: ------")
        print_weights(model_weights, discocirc_trainer.model_class.weights)

    accuracy = discocirc_trainer.get_accuracy(discocirc_trainer.dataset)
    print("The accuracy on the train set is", accuracy)

    if training_config["log_wandb"]:
        wandb.log({"train_accuracy": accuracy})

    if output_config['save_model']:
        save_base_path = base_path + save_path + model_class.__name__
        Path(save_base_path).mkdir(parents=True, exist_ok=True)
        name = save_base_path + "/" + model_class.__name__ + "_" \
               + datetime.utcnow().strftime("%h_%d_%H_%M")

        discocirc_trainer.save(name, save_traces=False)

        shutil.make_archive(name, 'zip', name)

        if training_config["log_wandb"]:
            wandb.save(name + '.zip')


if __name__ == "__main__":
    train(base_path,
          "/saved_models/",
          '/data/task_vocab_dicts/',
          "/data/pickled_dataset/")
