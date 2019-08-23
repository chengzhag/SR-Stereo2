from .SR import SR
from .EDSR import EDSR
from .SRdisp import SRdisp
from .MDSR import MDSRfeature
from .EDSR import EDSRfeature
from .PSMNetSR import *
from ..Stereo.PSMNet import PSMNetFeature
from .Interpolation import Bilinear

def getMask(model):
    if type(model) in (list, tuple):
        model = model[0]
    if model in ('EDSR', ) or 'PSMNetSR' in model:
        mask = (1, 1, 0, 0)
    elif model in ('SRdisp',):
        mask = (1, 1, 1, 1)
    else:
        raise Exception('Error: No model named \'%s\'!' % model)
    return mask


def getModel(model, cuda, half):
    if type(model) in (list, tuple):
        if model[0] in ('SRdisp',):
            if model[1] == 'EDSR':
                sr = globals()[model[1]](cInput=6, cuda=cuda, half=half)
            else:
                raise Exception(f'Error: No model {model[1]}')
            return globals()[model[0]](sr)
        if len(model) == 1:
            model = model[0]
    if 'PSMNetSR' in model:
        return getPSMNetSR(globals()['Raw' + model])(cuda=cuda, half=half)
    return globals()[model](cuda=cuda, half=half)
