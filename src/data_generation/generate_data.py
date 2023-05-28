import os, sys

import pickle

from discopy.monoidal import Box, Ty

from data_generation.generate_answer_pair_number import find_wire
from discocirc.helpers.discocirc_utils import get_star_removal_functor
from discocirc.pipeline.text_to_circuit import sentence_list_to_circuit

TASK_BASE_PATH = '/data/tasks_1-20_v1-2/en/'
SAVE_BASE_PATH = '/data/'
# want p = absolute path to \Neural-DisCoCirc
p = os.path.abspath('../..')
# p = os.path.abspath('..')
animal_dict = {'cat': 'cats', 'wolf': 'wolves', 'mouse': 'mice', 'sheep': 'sheep'}
task_specifics = {
    #==================== Task 1 ================================
    # 1 John travelled to the hallway.
    # 2 Mary journeyed to the bathroom.
    # 3 Where is John ? 	hallway	1
    1: {'get_question': lambda q : [q.split()[-2]],
        'get_answer': lambda a: [a],
        'get_question_id': True,
        'get_answer_id': True},
    # ==================== Task 6 ================================
    # 1 Mary got the milk there.
    # 2 John moved to the bedroom.
    # 3 Is John in the kitchen ? 	no	2
    6: {'get_question': lambda q: [q.split()[1], q.split()[-2]],
        'get_answer': lambda a: [a],
        'get_question_id': True,
        'get_answer_id': False},
    #==================== Task 7 ================================
    # 1 Mary got the milk there.
    # 2 John moved to the bedroom.
    # 3 How many objects is Mary carrying ? 	one	1
    7: {'get_question': lambda q: [q.split()[-3]],
        'get_answer': lambda a: [a],
        'get_question_id': True,
        'get_answer_id': False},
    #==================== Task 9 =================================
    # 1 John is in the hallway.
    # 2 Sandra is in the kitchen.
    # 3 Is Sandra in the bedroom ? 	no	2
    9: {'get_question': lambda q: [q.split()[1], q.split()[-2]],
        'get_answer': lambda a: [a],
        'get_question_id': True,
        'get_answer_id': False},
    #==================== Task 10 ================================
    # 1 Fred is either in the school or the park.
    # 2 Mary went back to the office.
    # 3 Is Mary in the office ? 	yes	2
    10: {'get_question': lambda q: [q.split()[1], q.split()[-2]],
        'get_answer': lambda a: [a],
        'get_question_id': True,
        'get_answer_id': False},
    #==================== Task 12 ================================
    # 1 John and Mary travelled to the hallway.
    # 2 Sandra and Mary journeyed to the bedroom.
    # 3 Where is Mary ? 	bedroom	2
    12: {'get_question': lambda q: [q.split()[-2]],
        'get_answer': lambda a: [a],
        'get_question_id': True,
        'get_answer_id': True},
    #==================== Task 15 ================================
    # 1 Wolves are afraid of mice.
    # ...
    # 8 Gertrude is a wolf.
    # 9 What is emily afraid of ?	wolf	7 5
    15: {'get_question': lambda q: [q.split()[-4]],
        'get_answer': lambda a: [a], # has to be plural as otherwise the answer may not yet have appeard in the context
        'get_question_id': True,
        'get_answer_id': True},

}

# read the .txt file
def task_file_reader(path):
    """
    reads the .txt file at path
    returns 3 lists of equal length
    - context sentences, questions, and answers
    """
    with open(path) as f:
        lines = f.readlines()


    # split the lines into stories
    # record the first line location of new stories
    story_splits = [i for i, line in enumerate(lines) if line[0:2] == '1 ']
    # have no more need for line indices - delete these
    lines = [' '.join(line.split(' ')[1:]) for line in lines]
    # also delete . and \n
    lines = [line.replace('.', '').replace('\n','') for line in lines]
    stories = [lines[i:j] for i, j in zip(story_splits, story_splits[1:]+[None])]

    # create context and QnA pairs
    contexts = []
    qnas = []
    for story in stories:
        # record the lines in the story corresponding to questions
        question_splits = [i for i, line in enumerate(story) if '?' in line]
        for index in question_splits:
            # record the context corresponding to each question
            contexts.append([line.lower() for line in story[:index] if '?' not in line])
            # record the question
            qnas.append(story[index])


    # split qna into questions and answers
    questions = [qna.split('\t')[0].lower()[:-1] + " ?" for qna in qnas]
    answers = [qna.split('\t')[1].lower() for qna in qnas]
    return contexts, questions, answers

def generate_data(task_file, task_specifics):
    star_removal_functor = get_star_removal_functor()
    contexts, questions, answers = task_file_reader(p + task_file)

    contexts = contexts[:20]
    questions = questions[:20]
    answers = answers[:20]

    dataset = [{} for _ in range(len(contexts))]
    vocab = []

    for i, context in enumerate(contexts):
        if task_specifics['get_question_id']:
            context += task_specifics['get_question'](questions[i])

        if task_specifics['get_answer_id']:
            context += task_specifics['get_answer'](answers[i])

        print(context)
        context_circ = star_removal_functor(
            sentence_list_to_circuit(context, simplify_swaps=False,
                                                wire_order='intro_order')
        )

        # context_circ.draw()
        dataset[i]['context_circ'] = context_circ

        question_circ = star_removal_functor(
            sentence_list_to_circuit([questions[i]], simplify_swaps=False,
                                                wire_order='intro_order')
        )
        dataset[i]['question_circ'] = question_circ

        for box in context_circ.boxes + question_circ.boxes:
            if box not in vocab:
                print(box.name, box.dom, box.cod)
                vocab.append(box)

        dataset[i]['question'] = task_specifics['get_question'](questions[i])
        dataset[i]['answer'] = task_specifics['get_answer'](answers[i])

        for name in dataset[i]['question'] + dataset[i]['answer']:
            box = Box(name, Ty(), Ty('n'))
            if box not in vocab:
                vocab.append(box)

        if task_specifics['get_question_id']:
            dataset[i]['question_id'] = \
                [find_wire(context_circ, q) for q in dataset[i]['question']]

        if task_specifics['get_answer_id']:
            dataset[i]['answer_id'] = \
                [find_wire(context_circ, a) for a in dataset[i]['answer']]

        if i % 10 == 0:
            print('finished context {} out of {}'.format(i, len(contexts)))

    return dataset, vocab

def generate(task_file, task_number, task_type):
    dataset_save_file = "/data/pickled_dataset/task{:02d}_{}.p".format(number, type)
    vocab_save_file = "/data/task_vocab_dicts/task{:02d}_{}.p".format(number, type)

    dataset, vocab = generate_data(task_file, task_specifics[task_number])

    with open(p + dataset_save_file, "wb") as f:
        pickle.dump(dataset, f)

    with open(p + vocab_save_file, "wb") as f:
        pickle.dump(vocab, f)


if __name__ == "__main__":
    for filename in sorted(os.listdir(p + TASK_BASE_PATH)):
        number = int(filename.split("_")[0][2:])
        type = "test" if "test" in filename else "train"


        # if number not in [15]:
        # if number not in [1, 6, 7, 9, 10, 12, 15]:
        if number not in [15]:
            continue
        if type == 'test':
            continue
        # if os.path.isfile(p + SAVE_BASE_PATH + save_file):
        #     print("skipping because save file already exists: {}".format(filename))
        #     continue

        # try:
        generate(TASK_BASE_PATH + filename, number, type)
        # except Exception as e:
        #     print("skipping due to error {}".format(filename))
        #     print(e)