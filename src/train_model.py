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

from network.utils.callbacks import ValidationAccuracy, \
    ModelCheckpointWithoutSaveTraces
from sklearn.model_selection import train_test_split

# this should the path to \Neural-DisCoCirc
base_path = os.path.abspath('..')
# base_path = os.path.abspath('.')
config = {
    "batch_size": 32,
    "dataset": "task1_train_dataset.pkl",
    "epochs": 20,
    "learning_rate": 0.001,
    "log_wandb": False,
    "model": LSTMModel,
    # "trainer": OneNetworkTrainer,
    "trainer": IndividualNetworksTrainer,
    "lexicon": "en_qa1.p",
}

all_configs = {
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


def train(base_path, save_path, vocab_path,
          data_path):
    model_class = config['model']

    print('Training: {} with trainer {} on data {}'
          .format(model_class.__name__,
                  config['trainer'].__name__,
                  config["dataset"]))

    print('loading vocabulary...')
    with open(base_path + vocab_path + config["lexicon"], 'rb') as file:
        lexicon = pickle.load(file)


    print('create model_config')
    model_config = {}
    for val in signature(model_class.__init__).parameters:
        if val not in all_configs.keys():
            continue
        model_config[val] = all_configs[val]

    config.update(model_config)
    if config["log_wandb"]:
        wandb.init(project="discocirc", entity="domlee", config=config)

    print('initializing model...')

    discocirc_trainer = config['trainer'](lexicon=lexicon, model_class=model_class, hidden_layers=all_configs['hidden_layers'], **model_config)

    print('loading pickled dataset...')
    with open(base_path + data_path + config['dataset'],
              "rb") as f:
        # dataset is a tuple (context_circuit,(question_word_index, answer_word_index))
        dataset = pickle.load(f)[:20]

    train_dataset, validation_dataset = train_test_split(dataset,
                                                         test_size=0.1,
                                                         random_state=1)

    discocirc_trainer.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config["learning_rate"]),
        run_eagerly=True
    )

    datetime_string = datetime.now().strftime("%B_%d_%H_%M")

    tb_callback = keras.callbacks.TensorBoard(
        log_dir='logs/{}'.format(datetime_string),
        histogram_freq=0,
        write_graph=True,
        write_images=True,
        update_freq='batch',
    )

    save_base_path = base_path + "/checkpoints/"
    checkpoint_callback = ModelCheckpointWithoutSaveTraces(
        filepath='{}/{}'.format(save_base_path, datetime_string),
        save_freq=20 * config["batch_size"]
    )

    validation_callback = ValidationAccuracy(discocirc_trainer.get_accuracy,
                                             interval=1,
                                             log_wandb=config["log_wandb"])

    print('training...')

    callbacks = [tb_callback, validation_callback, checkpoint_callback]

    if config["log_wandb"]:
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

    if config["log_wandb"]:
        wandb.log({"train_accuracy": accuracy})

    save_base_path = base_path + save_path + model_class.__name__
    Path(save_base_path).mkdir(parents=True, exist_ok=True)
    name = save_base_path + "/" + model_class.__name__ + "_" \
           + datetime.utcnow().strftime("%h_%d_%H_%M")

    discocirc_trainer.save(name, save_traces=False)

    shutil.make_archive(name, 'zip', name)

    if config["log_wandb"]:
        wandb.save(name + '.zip')

if __name__ == "__main__":
    train(base_path,
          "/saved_models/",
          '/data/task_vocab_dicts/',
          "/data/pickled_dataset/")
