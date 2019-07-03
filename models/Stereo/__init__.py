from .PSMNet import PSMNet
from .GwcNet import GwcNetG, GwcNetGC
# from .GANet import GANet
from .StereoDown import StereoDown
from .SRStereo import SRStereo
from .SRdispStereo import SRdispStereo
from .SRStereoRefine import SRStereoRefine
from .. import SR
from .PSMNet import PSMNetBody
from .FeatureStereo import FeatureStereo

def getModel(model, kitti, maxDisp, dispScale, cuda, half):
    if type(model) in (list, tuple):
        if model[0] in ('StereoDown',):
            stereo = getModel(model[1], kitti=kitti, maxDisp=maxDisp, dispScale=dispScale, cuda=cuda, half=half)
            return globals()[model[0]](stereo)
        elif model[0] in ('SRStereo',):
            sr = SR.getModel(model[1], cuda=cuda, half=half)
            stereo = getModel(
                ('StereoDown', model[2]), kitti=kitti, maxDisp=maxDisp, dispScale=dispScale, cuda=cuda, half=half)
            return globals()[model[0]](sr, stereo)
        elif model[0] in ('SRdispStereo', 'SRStereoRefine'):
            sr = SR.getModel(
                ('SRdisp', model[1]), cuda=cuda, half=half)
            stereo = getModel(
                ('StereoDown', model[2]), kitti=kitti, maxDisp=maxDisp, dispScale=dispScale, cuda=cuda, half=half)
            return globals()[model[0]](sr.sr, stereo)
        elif model[0] in ('FeatureStereo'):
            feature = SR.getModel(model[1], cuda=cuda, half=half)
            stereoBody = globals()[model[2]](
                kitti=kitti, maxDisp=maxDisp, dispScale=dispScale, cuda=cuda, half=half, cFeature=feature.cOutput)
            return FeatureStereo(feature=feature, stereoBody=stereoBody)
        if len(model) == 1:
            model = model[0]
    return globals()[model](kitti=kitti, maxDisp=maxDisp, dispScale=dispScale, cuda=cuda, half=half)
