############################################################
# generate data for 'IsIn model' (tasks 1, 2)

############################################################


# path nonsense
import os, sys

# we want p = the absolute path to \src
p = os.path.abspath('..')
# p = os.path.abspath('./src')
print('PATH TO src ', p)
sys.path.insert(1, p)


# some instructions specific to task 1
import pickle

# from data_generation.generate_answer_pair_number import get_qa_numbers
# from data_generation.prepare_data_utils import task_file_reader
# from discocirc.discocirc_utils import get_star_removal_functor
# from discocirc.text_to_circuit import sentence_list_to_circuit

from data_generation.generate_answer_pair_number import get_qa_numbers
from data_generation.prepare_data_utils import task_file_reader
# from discocirc.discocirc_utils import get_star_removal_functor
# from discocirc.text_to_circuit import sentence_list_to_circuit
from discocirc.helpers.discocirc_utils import get_star_removal_functor
from discocirc.pipeline.text_to_circuit import sentence_list_to_circuit


TASK_FILE = '/data/tasks_1-20_v1-2/en/qa12_conjunction_train.txt'
# TASK_FILE = '/../../data/tasks_1-20_v1-2/en/qa12_conjunction_train.txt'
SAVE_FILE = '/data/pickled_dataset/add_logits_dataset_task12_train.pkl'
# SAVE_FILE = '/../../data/pickled_dataset/add_logits_dataset_task12_train.pkl'


context_file = '/src/data_generation/context_circuits_task12.txt'


# want p = absolute path to \Neural-DisCoCirc
p = os.path.abspath('../..')
# p = os.path.abspath('.')
print('PATH TO Neural-DisCoCirc: ', p)


print(p+context_file)
print(p+TASK_FILE)
print(p+SAVE_FILE)


contexts, questions, answers = task_file_reader(p+TASK_FILE)

# # smaller dataset for testing
# contexts = contexts[:20]
# questions = questions[:20]
# answers = answers[:20]


# import spacy
# load_model = spacy.load("en_core_web_sm", disable=['parser','ner'])
#
# outer = []
# for i in range(len(contexts)):
#     inner = []
#     for j in range(len(contexts[i])):
#         doc = load_model(contexts[i][j])
#         list = []
#         for token in doc:
#             list.append(token.lemma_)
#             string = " ".join(list)
#         inner.append(string)
#         print("inner", inner)
#     outer.append(inner)
#
#
# print("outer", outer)
#
# contexts = outer


# generate context circuits from context sentences
# context_circuits = []
# for i, context in enumerate(contexts):
#     # generate a circuit using all sentences in this example's context
#     context_circ = sentence_list_to_circuit(context, simplify_swaps=False, wire_order = 'intro_order')
#     context_circuits.append(context_circ)
#     print('finished context {}'.format(i), end='\r')

with open(p+context_file, "rb") as f:
    context_circuits = pickle.load(f)

star_removal_functor = get_star_removal_functor()
q_a_number_pairs = get_qa_numbers(context_circuits, questions, answers)
dataset = []
for circ, q_a_number_pair, question, answer in \
        zip(context_circuits, q_a_number_pairs, questions, answers):
    circ = star_removal_functor(circ)
    dataset.append((circ, (q_a_number_pair[0], answer)))


# STOP HERE
print('hello')


with open(p+SAVE_FILE, "wb") as f:
    pickle.dump(dataset, f)


# %%
