from .CausalClassifier import CausalClassifier, MLP


def select_model(model):

    if model == 'MLP':
        model = MLP
    else:
        model = MLP

    return model