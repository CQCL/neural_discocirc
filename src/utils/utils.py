import numpy as np
from discopy import PRO
from discopy.monoidal import Functor
from sklearn.metrics import accuracy_score
from tensorflow import keras

from utils.neural_network_category import Network


def create_feedforward_network(input_dim, output_dim, hidden_layers,
                               activation='relu', name=None):
    input = keras.Input(shape=(input_dim,))
    output = input
    for layer in hidden_layers:
        output = keras.layers.Dense(layer, activation=activation, bias_initializer="glorot_uniform")(output)
    # output = keras.layers.Dense(output_dim, bias_initializer="glorot_uniform")(output)
    #### ADD activation in last layer!!
    output = keras.layers.Dense(output_dim, activation=activation, bias_initializer="glorot_uniform")(output)
    return keras.Model(inputs=input, outputs=output, name=name)

def create_feedforward_network_binary(input_dim, output_dim, hidden_layers,
                               activation='relu', name=None):
    input = keras.Input(shape=(input_dim,))
    output = input
    for layer in hidden_layers:
        output = keras.layers.Dense(layer, activation=activation, bias_initializer="glorot_uniform")(output)
    # output = keras.layers.Dense(output_dim, bias_initializer="glorot_uniform")(output)
    #### ADD activation in last layer!!
    output = keras.layers.Dense(output_dim, activation="sigmoid", bias_initializer="glorot_uniform")(output)
    return keras.Model(inputs=input, outputs=output, name=name)


def make_lambda_layer(a, b):
    return keras.layers.Lambda(lambda x: x[:, a:b])


def make_swap_layer(wire_dim):
    inputs = keras.Input(shape=(2 * wire_dim,))
    model1 = keras.layers.Lambda(
        lambda x: x[:, :wire_dim], )(inputs)
    model2 = keras.layers.Lambda(
        lambda x: x[:, wire_dim:], )(inputs)
    outputs = keras.layers.Concatenate()([model2, model1])
    return keras.Model(inputs=inputs, outputs=outputs)


def get_fast_nn_functor(nn_boxes, wire_dim):
    from discopy.monoidal import Swap, Ty
    nn_boxes[repr(Swap(Ty('n'), Ty('n')))] = make_swap_layer(wire_dim)

    def fast_f(diagram):
        inputs = keras.Input(shape=(len(diagram.dom) * wire_dim))
        outputs = inputs
        for fol in diagram.foliation():
            in_idx = 0
            out_idx = 0
            models = []
            for left, box, right in fol.layers:
                n_wires = (in_idx + len(left) - out_idx) * wire_dim
                f_idx = in_idx * wire_dim
                if f_idx < n_wires:
                    model = make_lambda_layer(f_idx, n_wires)(outputs)
                    models.append(model)
                f_dom = len(box.dom) * wire_dim
                model = make_lambda_layer(n_wires, n_wires + f_dom)(outputs)
                models.append(nn_boxes[repr(box)](model))
                in_idx = len(left) - out_idx + len(box.dom)
                out_idx = len(left) + len(box.cod)
            if right:
                model = make_lambda_layer(n_wires + f_dom, None)(outputs)
                models.append(model)
            outputs = keras.layers.Concatenate()(models)
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model

    return fast_f


def initialize_boxes(lexicon, wire_dimension, hidden_layers=[10, ]):
    """
    Returns a dict of neural networks, and a list of models

    Parameters
    ----------
    lexicon : list
        List of discopy boxes in the lexicon.
    wire_dimension : int
        Dimension of the noun wires.
    """
    nn_boxes = {}
    for word in lexicon:
        name = word.name
        if '\\' in name:
            name = name.replace('\\', '')
            name = name[1:-1] + '_end'
        elif '[' in name:
            name = name[1:-1] + '_begin'
        nn_boxes[repr(word)] = create_feedforward_network(
            input_dim=len(word.dom) * wire_dimension,
            output_dim=len(word.cod) * wire_dimension,
            hidden_layers=hidden_layers,
            name=name
        )
    return nn_boxes


def get_classification_vocab(lexicon):
    """
    Parameters:
    -----------
    lexicon : list
        list of discopy boxes

    Returns:
    --------
    vocab : list
        list of names of boxes (modulo frames)
    """
    vocab = []
    for box in lexicon:
        name = box.name
        if '[' in name:
            name = name.replace('\\', '')
            name = name[1:-1]
        if name not in vocab:
            vocab.append(name)
    return vocab


def get_params_dict_from_tf_variables(params, split_string, is_state=False):
    params_dict = {}
    for p in params:
        name = p.name[:-2].split(split_string)[0]
        if is_state:
            params_dict[name] = p
        else:
            if name in params_dict:
                params_dict[name].append(p)
            else:
                params_dict[name] = [p]
    return params_dict


def get_box_name(box):
    name = box.name
    if '\\' in name:
        name = name.replace('\\', '')
        name = name[1:-1] + '_end'
    elif '[' in name:
        name = name[1:-1] + '_begin'
    name = name + '_' + str(len(box.dom)) + '_' + str(len(box.cod))
    return name
