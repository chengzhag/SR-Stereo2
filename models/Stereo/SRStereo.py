import torch.optim as optim
import torch
import torch.nn as nn
from utils import myUtils
import collections
from .Stereo import Stereo
from .. import SR
from evaluation import evalFcn


class RawSRStereo(nn.Module):
    def __init__(self, sr: SR.SR, stereo: Stereo):
        super().__init__()
        def getModel(model):
            model = stereo.model
            if hasattr(model, 'module'):
                model = model.module
            return model
        self.sr = getModel(sr)
        self.stereo = getModel(stereo)


class SRStereo(Stereo):

    def __init__(self, sr: SR.SR, stereo: Stereo):
        super().__init__(
            maxDisp=stereo.maxDisp, dispScale=stereo.dispScale, cuda=stereo.cuda, half=stereo.half)
        stereo.optimizer = None
        sr.optimizer = None
        self.stereo = stereo
        self.sr = sr
        self.initModel()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=0.001, betas=(0.9, 0.999)
        )

    def initModel(self):
        self.model = RawSRStereo(self.sr, self.stereo)

    def predict(self, batch: myUtils.Batch, mask=(1, 1)):
        batch.assertScales(1)
        outputs = self.sr.predict(batch=batch)
        batch.lowestResRGBs((outputs['outputSrL'], outputs['outputSrR']))
        outputs.update(self.stereo.predict(batch=batch, mask=mask))
        return outputs

    def load(self, checkpointDir):
        if checkpointDir is None:
            return None, None

        if type(checkpointDir) in (list, tuple) and len(checkpointDir) == 2:
            # Load pretrained SR and Stereo weights
            self.sr.load(checkpointDir[0])
            self.stereo.load(checkpointDir[1])
            return None, None
        elif type(checkpointDir) is str:
            # Load fintuned SRStereo weights
            return super(SRStereo, self).load(checkpointDir)
        else:
            raise Exception('Error: SRStereo need 2 checkpoints SR/Stereo or 1 checkpoint SRStereo to load!')
