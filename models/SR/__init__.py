from .SR import SR
from .EDSR import EDSR
from .EDSRdisp import EDSRdisp


def getMask(model):
    if type(model) in (list, tuple):
        assert len(model) == 1
        model = model[0]
    if model in ('EDSR',):
        mask = (1, 1, 0, 0)
    elif model in ('EDSRdisp',):
        mask = (1, 1, 1, 1)
    else:
        raise Exception('Error: No model named \'%s\'!' % model)
    return mask

def getModel(model, cuda, half):
    if type(model) in (list, tuple):
        assert len(model) == 1
        model = model[0]
    return globals()[model](cuda=cuda, half=half)

