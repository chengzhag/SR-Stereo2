from .PSMNet import PSMNet
from .PSMNetDown import PSMNetDown
from .SRStereo import SRStereo
# from .SRdispStereo import SRdispStereo
# from .SRdispStereoRefine import SRdispStereoRefine
from .. import SR


def getModel(model, maxDisp, dispScale, cuda, half):
    if type(model) in (list, tuple) and len(model) == 3:
        sr = SR.getModel(model[1], cuda=cuda, half=half)
        stereo = getModel(model[2], maxDisp=maxDisp, dispScale=dispScale, cuda=cuda, half=half)
        return globals()[model[0]](sr, stereo)
    if type(model) in (list, tuple) and len(model) == 1:
        model = model[0]
    return globals()[model](maxDisp, dispScale, cuda=cuda, half=half)


