from .CausalClassifier import CausalClassifier
from .model_backbone import MLP, CLIPMLP


def select_model(model):

    if model == 'MLP':
        model = MLP
    elif model == 'CLIPMLP':
        model = CLIPMLP
    else:
        model = MLP

    return model