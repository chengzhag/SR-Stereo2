from .EDSR import EDSR
from .EDSRdisp import EDSRdisp

def getMask(model):
    if model in ('EDSR',):
        mask = (1, 1, 0, 0)
    elif model in ('EDSRdisp',):
        mask = (1, 1, 1, 1)
    else:
        raise Exception('Error: No model named \'%s\'!' % model)
    return mask
