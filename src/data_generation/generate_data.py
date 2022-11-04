import os, sys

import pickle

from data_generation.generate_answer_pair_number import find_wire
from data_generation.prepare_data_utils import task_file_reader
from discocirc.discocirc_utils import get_star_removal_functor
from discocirc.text_to_circuit import sentence_list_to_circuit

TASK_BASE_PATH = '/data/tasks_1-20_v1-2/en/'
SAVE_BASE_PATH = '/data/pickled_dataset/'
# want p = absolute path to \Neural-DisCoCirc
p = os.path.abspath('../..')
# p = os.path.abspath('..')


# %%
star_removal_functor = get_star_removal_functor()


def generate(task_file, save_file):
    print(task_file, save_file)
    print('PATH TO Neural-DisCoCirc: ', p)
    contexts, questions, answers = task_file_reader(p + task_file)

    dataset = [{} for _ in range(len(contexts))]
    for i, context in enumerate(contexts):
        context_circ = sentence_list_to_circuit(context, simplify_swaps=False,
                                                wire_order='intro_order')
        dataset[i]['context_circ'] = star_removal_functor(context_circ)

        question_circ = sentence_list_to_circuit([questions[i][:-1]])
        dataset[i]['question_circ'] = star_removal_functor(question_circ)

        dataset[i]['question'] = questions[i]
        dataset[i]['answer'] = answers[i]

        dataset[i]['question_id'] = find_wire(context_circ,
                                              questions[i].split()[-1][:-1])
        dataset[i]['answer_id'] = find_wire(context_circ, answers[i])

        print('finished context {}'.format(i))

    with open(p + save_file, "wb") as f:
        pickle.dump(dataset, f)


if __name__ == "__main__":
    for filename in sorted(os.listdir(p + TASK_BASE_PATH)):
        number = int(filename.split("_")[0][2:])
        type = "test" if "test" in filename else "train"
        save_file = "task{:02d}_{}_dataset.pkl".format(number, type)
        if not os.path.isfile(p + SAVE_BASE_PATH + save_file):
            # TODO: add p to path and remove from function
            try:
                generate(TASK_BASE_PATH + filename, SAVE_BASE_PATH + save_file)
            except:
                print("skipping {}".format(filename))


# %%
