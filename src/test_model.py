import os

import numpy as np
from pandas import DataFrame

import pickle
from tensorflow import keras

from network.big_network_models.is_in_one_network import \
    IsInOneNetworkTrainer
from network.big_network_models.one_network_trainer_base import OneNetworkTrainer
from network.individual_networks_models.individual_networks_trainer_base_class import \
    IndividualNetworksTrainer
from network.individual_networks_models.is_in_trainer import \
    IsInIndividualNetworksTrainer

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# this should the the path to \Neural-DisCoCirc
# base_path = os.path.abspath('..')
base_path = os.path.abspath('.')

config = {
    "trainer": OneNetworkTrainer,
    "model_class": IsInOneNetworkTrainer,
    "dataset": "isin_dataset_task1_test.pkl",
    "vocab": "en_qa1.p",
    "model": "IsInOneNetworkTrainer/IsInOneNetworkTrainer_Oct_12_16_36"
}

def create_answer_dataframe(discocirc_trainer, vocab_dict, dataset):
    df = DataFrame([],
                   columns=['answer', 'correct', 'person', 'person_wire_no'])
    for i, (context_circuit_model, test) in enumerate(
            discocirc_trainer.dataset):
        person, location = test

        answer_prob = discocirc_trainer.call((context_circuit_model, person))
        answer_id = np.argmax(answer_prob)

        given_answer = list(vocab_dict.keys())[
                           list(vocab_dict.values()).index(answer_id)],
        correct_answer_name = dataset[i][0][person].boxes[0].name

        print("answer: {}, correct: {}, person: {}, {}".format(
            given_answer, location, person, correct_answer_name))

        df.loc[len(df.index)] = [
            given_answer, location, person, correct_answer_name]

    df.to_csv("answers.csv")


def test(base_path, model_path, vocab_path, test_path):
    model_base_path = base_path + model_path + config["model"]
    test_base_path = base_path + test_path + config["dataset"]

    trainer_class = config["trainer"]

    print('Testing: {} from path {} with data {}'
          .format(trainer_class.__name__, model_base_path, test_base_path))

    print('loading vocabulary...')
    with open(base_path + vocab_path + config["vocab"], 'rb') as file:
        lexicon = pickle.load(file)

    print('initializing model...')
    discocirc_trainer = trainer_class.load_model(model_base_path, config['model_class'])

    print('loading pickled dataset...')
    with open(test_base_path, "rb") as f:
        dataset = pickle.load(f)[:5]

    print('compiling dataset (size: {})...'.format(len(dataset)))

    discocirc_trainer.compile(optimizer=keras.optimizers.Adam(),
                              run_eagerly=True)

    accuracy = trainer_class.get_accuracy(discocirc_trainer, dataset)

    print("The accuracy on the test set is", accuracy)

    # create_answer_dataframe(discocirc_trainer, vocab_dict, dataset)


if __name__ == "__main__":
    test(base_path,
         "/saved_models/",
         '/data/task_vocab_dicts/',
         "/data/pickled_dataset/")
