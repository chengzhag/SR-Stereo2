import torch
import torch.nn as nn
import utils.data
import utils.experiment
import utils.imProcess
from .Stereo import Stereo
import torch.optim as optim

from .PSMNet import gerRawPSMNetScale
from .RawGwcNet import GwcNet_G as rawGwcNetG
from .RawGwcNet import GwcNet_GC as  rawGwcNetGC
from .RawGwcNet import model_loss as lossGwcNet

class GwcNet(Stereo):
    def __init__(self, kitti, maxDisp=192, dispScale=1, cuda=True, half=False, rawGwcNet=None):
        super().__init__(kitti=kitti, maxDisp=maxDisp, dispScale=dispScale, cuda=cuda, half=half)
        self.rawGwcNet = rawGwcNet
        self.initModel()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, betas=(0.9, 0.999))
        if self.cuda:
            self.model = nn.DataParallel(self.model)
            self.model.cuda()

    def initModel(self):
        self.model = gerRawPSMNetScale(self.rawGwcNet)(maxDisp=self.maxDisp, dispScale=self.dispScale)

    def packOutputs(self, outputs: dict, imgs: utils.imProcess.Imgs = None) -> utils.imProcess.Imgs:
        imgs = super().packOutputs(outputs, imgs)
        for key, value in outputs.items():
            if key.startswith('outputDisp'):
                if type(value) in (list, tuple):
                    value = value[3].detach()
                imgs.addImg(name=key, img=value, range=self.outMaxDisp)
        return imgs

    # input disparity maps:
    #   disparity range: 0~self.maxdisp * self.dispScale
    #   format: NCHW
    def loss(self, output: utils.imProcess.Imgs, gt: torch.Tensor, outMaxDisp=None):
        if outMaxDisp is None:
            outMaxDisp = self.outMaxDisp
        # for kitti dataset, only consider loss of none zero disparity pixels in gt
        loss = utils.data.NameValues()
        mask = ((gt < outMaxDisp) & (gt > 0)).detach()
        loss['lossDisp'] = lossGwcNet(output['outputDisp'], gt, mask)
        loss['loss'] = loss['lossDisp'] * self.lossWeights

        return loss

    def train(self, batch: utils.data.Batch, progress=0):
        return self.trainBothSides(batch.oneResRGBs(), batch.oneResDisps())

def GwcNetG(kitti, maxDisp=192, dispScale=1, cuda=True, half=False):
    return GwcNet(kitti=kitti, maxDisp=maxDisp, dispScale=dispScale, cuda=cuda, half=half, rawGwcNet=rawGwcNetG)

def GwcNetGC(kitti, maxDisp=192, dispScale=1, cuda=True, half=False):
    return GwcNet(kitti=kitti, maxDisp=maxDisp, dispScale=dispScale, cuda=cuda, half=half, rawGwcNet=rawGwcNetGC)
