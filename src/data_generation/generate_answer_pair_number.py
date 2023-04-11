from discopy import Ty
from distutils.util import strtobool


def find_wire(diagram, name):
    index = 0
    box = diagram.boxes[index]
    while box.dom == Ty():
        if box.name == name:
            return diagram.offsets[index]

        index += 1
        box = diagram.boxes[index]

    raise Exception('wire not found')


def get_qa_numbers(context_circuits, questions, answers):
    """
    Parameters:
    -----------
    context_circuits : list
        List of context circuits.
    questions : list
        List of strings
    answers : list
        List of strings
    
    Returns:
    --------
    q_a_number_pairs : list
        List of tuples (question word index, answer word index)
    """
    q_a_pairs = []
    # the following is quite hard-coded
    for question, answer_word in zip(questions, answers):
        # the 'question word' for tasks 1, 2 is always the last word in the question
        question_word = question.split()[-1][:-1]
        q_a_pairs.append((question_word, answer_word))

    q_a_number_pairs = []
    for i, (question, answer) in enumerate(q_a_pairs):
        q_id = find_wire(context_circuits[i], question)
        a_id = find_wire(context_circuits[i], answer)
        q_a_number_pairs.append((q_id, a_id))

    return q_a_number_pairs


def get_qa_numbers_task15(context_circuits, questions, answers):
    """
    Parameters:
    -----------
    context_circuits : list
        List of context circuits.
    questions : list
        List of strings
    answers : list
        List of strings

    Returns:
    --------
    q_a_number_pairs : list
        List of tuples (question word index, answer word index)
    """
    q_a_pairs = []
    # the following is quite hard-coded
    for question, answer_word in zip(questions, answers):
        # the 'question word' for tasks 15 is always the third word in the question
        # question_word = question.split()[-1][:-1]
        question_word = question.split()[2]
        q_a_pairs.append((question_word, answer_word))

    q_a_number_pairs = []
    for i, (question, answer) in enumerate(q_a_pairs):
        q_id = find_wire(context_circuits[i], question)
        a_id = find_wire(context_circuits[i], answer)
        q_a_number_pairs.append((q_id, a_id))

    return q_a_number_pairs

def get_qa_numbers_task6_addlogits(context_circuits, questions, answers):
    """
    Parameters:
    -----------
    context_circuits : list
        List of context circuits.
    questions : list
        List of strings
    answers : list
        List of strings

    Returns:
    --------
    q_a_number_pairs : list
        List of tuples (question word index, answer word index)
    """
    q_a_pairs = []
    # the following is quite hard-coded
    for question, answer_word in zip(questions, answers):
        # the 'question word' for task 6, person in second place, loc is always the last word in the question
        question_word_person = question.split()[1]
        question_word_location = question.split()[-1][:-1]
        question_word = (question_word_person, question_word_location)
        # answer_word = strtobool(answer_word)
        q_a_pairs.append((question_word, answer_word))

    q_a_number_pairs = []
    for i, (question, answer) in enumerate(q_a_pairs):
        q_person_id = find_wire(context_circuits[i], question[0])
        q_location_id = find_wire(context_circuits[i], question[1])
        q_id = (q_person_id, q_location_id)
        # a_id = find_wire(context_circuits[i], answer)
        # answer if not wire number as before, it is just yes/no
        q_a_number_pairs.append((q_id, answer))

    return q_a_number_pairs

def get_qa_numbers_task6_isin(context_circuits, questions, answers):
    """
    Parameters:
    -----------
    context_circuits : list
        List of context circuits.
    questions : list
        List of strings
    answers : list
        List of strings

    Returns:
    --------
    q_a_number_pairs : list
        List of tuples (question word index, answer word index)
    """
    q_a_pairs = []
    # the following is quite hard-coded
    for question, answer_word in zip(questions, answers):
        # the 'question word' for task 6, person in second place, loc is always the last word in the question
        question_word_person = question.split()[1]
        question_word_location = question.split()[-1][:-1]
        question_word = (question_word_person, question_word_location)
        answer_word = strtobool(answer_word)
        q_a_pairs.append((question_word, answer_word))

    q_a_number_pairs = []
    for i, (question, answer) in enumerate(q_a_pairs):
        q_person_id = find_wire(context_circuits[i], question[0])
        q_location_id = find_wire(context_circuits[i], question[1])
        q_id = (q_person_id, q_location_id)
        # a_id = find_wire(context_circuits[i], answer)
        # answer if not wire number as before, it is just yes/no
        q_a_number_pairs.append((q_id, answer))

    return q_a_number_pairs
