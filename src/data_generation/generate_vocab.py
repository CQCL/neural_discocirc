# %%
from discocirc.pipeline.text_to_circuit import sentence_list_to_circuit

import os
import sys
p = os.path.abspath('..') # this should be the path to \src
print("THE PATH TO src: ", p)
sys.path.insert(1, p)

from discocirc.helpers.discocirc_utils import get_star_removal_functor
import pickle   # for saving vocab file

def get_vocab_from_lines(lines):
    functor = get_star_removal_functor()
    vocab = []
    for i, line in enumerate(lines):

        # obtain circ for the line
        # print(line.lower())
        line_circ = sentence_list_to_circuit([line.lower()],
                                             simplify_swaps=False,
                                             wire_order='intro_order')

        # apply the star removal functor
        line_circ = functor(line_circ)

        line_boxes = line_circ.boxes

        for box in line_boxes:
            if box not in vocab:
                vocab.append(box)

        if i % 50 == 0:
            print("{} of {}".format(i, len(lines)))
            print("vocab size = {}".format(len(vocab)))

    return vocab

def get_vocab_from_file(all_lines):
    # delete initial line numbers
    all_lines = [' '.join(line.split(' ')[1:]) for line in
                     all_lines]

    # filter out the lines involving questions for now
    context_lines = [line for line in all_lines if '?' not in line]
    question_lines = [line for line in all_lines if '?' in line]

    # delete . and \n
    context_lines = [line.replace('\n', '').replace('.', ' ') for line in
                         context_lines]

    questions = [line.split('\t')[0].split('?')[0] for line in question_lines]
    answers = [line.split('\t')[1] for line in question_lines]

    return get_vocab_from_lines(context_lines + questions + answers)




def run():
    #%%
    # read the file
    base_path = os.path.abspath('../../data/')
    data_path = base_path + "/tasks_1-20_v1-2/en/"
    print("THE PATH TO Neural-DisCoCirc: ", data_path)

    vocab = []
    for filename in sorted(os.listdir(data_path)):
        if "qa1_" not in filename:
            continue

        print(filename)
        with open(data_path + "/" + filename) as f:
            lines = f.readlines()

        file_vocab = get_vocab_from_file(lines)
        for v in file_vocab:
            if v not in vocab:
                vocab.append(v)

        print("===== File complete. Vocab size: {} =====".format(len(vocab)))

    print(vocab)
    # save vocab file
    pickle.dump(vocab, open(base_path+"/task_vocab_dicts/en_qa1.p", "wb"))


if __name__ == '__main__':
    run()
# %%
