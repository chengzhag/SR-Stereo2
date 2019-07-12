import utils.experiment
import torch
import torch.nn.functional as F
import torch.nn as nn
import utils.data
import utils.imProcess
from .Stereo import Stereo
import torch.optim as optim
from apex import amp

from .PSMNet import getRawPSMNetScale
from .RawGANet import GANet as rawGANet
from .RawGANet.libs.GANet.modules.GANet import MyLoss2

class GANet(Stereo):
    def __init__(self, kitti, maxDisp=192, dispScale=1, cuda=True, half=False):
        super().__init__(kitti=kitti, maxDisp=maxDisp, dispScale=dispScale, cuda=cuda, half=half)
        self.initModel()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, betas=(0.9, 0.999))
        if self.cuda:
            self.model.cuda()
            self.model, self.optimizer = amp.initialize(models=self.model, optimizers=self.optimizer, enabled=half)
            self.model = nn.DataParallel(self.model)

    def initModel(self):
        self.model = getRawPSMNetScale(rawGANet)(maxDisp=self.maxDisp, dispScale=self.dispScale)
        self.model.multiple = 48
        self.showParamNum()

    def packOutputs(self, outputs: dict, imgs: utils.imProcess.Imgs = None) -> utils.imProcess.Imgs:
        imgs = super().packOutputs(outputs, imgs)
        for key, value in outputs.items():
            if key.startswith('outputDisp'):
                if type(value) in (list, tuple):
                    value = value[2].detach()
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

        if self.kitti:
            loss['lossDisp'] = 0.2 * F.smooth_l1_loss(output['outputDisp'][0][mask], gt[mask], reduction='mean') \
                   + 0.6 * F.smooth_l1_loss(output['outputDisp'][1][mask], gt[mask], reduction='mean') \
                   + MyLoss2(thresh=3, alpha=2)(output['outputDisp'][2][mask], gt[mask])
        else:
            loss['lossDisp'] = 0.2 * F.smooth_l1_loss(output['outputDisp'][0][mask], gt[mask], reduction='mean') \
                   + 0.6 * F.smooth_l1_loss(output['outputDisp'][1][mask], gt[mask], reduction='mean') \
                   + F.smooth_l1_loss(output['outputDisp'][2][mask], gt[mask],reduction='mean')

        loss['loss'] = loss['lossDisp'] * self.lossWeights

        return loss

    def train(self, batch: utils.data.Batch, progress=0):
        return self.trainBothSides(batch.oneResRGBs(), batch.oneResDisps())
