#%%
import sys
sys.setrecursionlimit(10000)
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from data_generation.generate_answer_pair_number import get_qa_numbers
from discocirc.discocirc_utils import get_star_removal_functor
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score

from network.model import DisCoCircTrainer
#%%

print('initializing model...')
discocirc_trainer = DisCoCircTrainer.load_models('./saved_models/trained_model_boxes.pkl')

print('loading pickled dataset...')
with open("data/discocirc_diagrams/context_circuits_test.pkl", "rb") as f:
    context_circuits = pickle.load(f)
q_a_number_pairs = get_qa_numbers(task_file='data/tasks_1-20_v1-2/en/qa1_single-supporting-fact_test.txt',
                   context_circuits="data/discocirc_diagrams/context_circuits_test.pkl")

#%%
star_removal_functor = get_star_removal_functor()
dataset = []
for circ, test in zip(context_circuits, q_a_number_pairs):
    circ = star_removal_functor(circ)
    dataset.append((circ, test))

#%%
print('compiling dataset')
discocirc_trainer.compile_dataset(dataset)
discocirc_trainer.compile(optimizer=keras.optimizers.Adam(), run_eagerly=True)
discocirc_trainer(0)


#%%
location_predicted = []
location_true = []
for i in range(len(dataset)):
    print('predicting {} / {}'.format(i, len(dataset)), end='\r')
    probs = discocirc_trainer(i)
    location_predicted.append(np.argmax(probs))
    location_true.append(q_a_number_pairs[i][1])


# %%
print(accuracy_score(location_true, location_predicted))
# %%