import torch
import torch.nn.functional as F
import torch.nn as nn
from utils import myUtils
from .RawPSMNet import stackhourglass as rawPSMNet
import collections
from .Stereo import Stereo
import torch.optim as optim


class RawPSMNetScale(rawPSMNet):
    def __init__(self, maxDisp, dispScale):
        super(RawPSMNetScale, self).__init__(maxDisp, dispScale)
        self.multiple = 16
        self.__imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                                 'std': [0.229, 0.224, 0.225]}

    # input: RGB value range 0~1
    # outputs: disparity range 0~self.maxdisp * self.dispScale
    def forward(self, left, right) -> myUtils.Output:
        def normalize(nTensor):
            nTensorClone = nTensor.clone()
            for tensor in nTensorClone:
                for t, m, s in zip(tensor, self.__imagenet_stats['mean'], self.__imagenet_stats['std']):
                    t.sub_(m).div_(s)
            return nTensorClone

        left, right = normalize(left), normalize(right)

        if self.training:
            rawOutputs = super(RawPSMNetScale, self).forward(left, right)
        else:
            autoPad = myUtils.AutoPad(left, self.multiple)

            left, right = autoPad.pad((left, right))
            rawOutputs = super(RawPSMNetScale, self).forward(left, right)
            rawOutputs = autoPad.unpad(rawOutputs)
        outputs = myUtils.Output()
        outputs.addDisp(rawOutputs, maxDisp=self.maxdisp)
        return outputs

    def load_state_dict(self, state_dict, strict=False):
        writeModelDict = self.state_dict()
        selectModelDict = {}
        for name, value in state_dict.items():
            possiblePrefix = 'stereo.module.'
            if name.startswith(possiblePrefix):
                name = name[len(possiblePrefix):]
            if name in writeModelDict and writeModelDict[name].size() == value.size():
                selectModelDict[name] = value
            else:
                message = 'Warning! While copying the parameter named {}, ' \
                          'whose dimensions in the model are {} and ' \
                          'whose dimensions in the checkpoint are {}.' \
                    .format(
                    name, writeModelDict[name].size() if name in writeModelDict else '(Not found)',
                    value.size()
                )
                if strict:
                    raise Exception(message)
                else:
                    print(message)
        writeModelDict.update(selectModelDict)
        super(RawPSMNetScale, self).load_state_dict(writeModelDict, strict=False)


class PSMNet(Stereo):
    def __init__(self, maxDisp=192, dispScale=1, cuda=True, half=False):
        super().__init__(maxDisp=maxDisp, dispScale=dispScale, cuda=cuda, half=half)
        self.initModel()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, betas=(0.9, 0.999))
        if self.cuda:
            self.model = nn.DataParallel(self.model)
            self.model.cuda()

    def initModel(self):
        self.model = RawPSMNetScale(maxDisp=self.maxDisp, dispScale=self.dispScale)

    # # input disparity maps:
    # #   disparity range: 0~self.maxdisp * self.dispScale
    # #   format: NCHW
    # def loss(self, outputs: tuple, gts: torch.Tensor, outMaxDisp=None, kitti=False):
    #     if outMaxDisp is None:
    #         outMaxDisp = self.outMaxDisp
    #     # for kitti dataset, only consider loss of none zero disparity pixels in gt
    #     mask = (gts > 0).detach() if kitti else (gts < outMaxDisp).detach()
    #     loss = myUtils.Loss()
    #     loss.addLoss(0.5 * F.smooth_l1_loss(outputs[0][mask], gts[mask], reduction='mean') + 0.7 * F.smooth_l1_loss(
    #         outputs[1][mask], gts[mask], reduction='mean') + F.smooth_l1_loss(outputs[2][mask], gts[mask],
    #                                                                           reduction='mean'),
    #                  name='Disp')
    #
    #     return loss

    # def trainOneSide(self, imgL, imgR, gt, kitti=False):
    #     self.optimizer.zero_grad()
    #     outputs = self.model.forward(imgL, imgR)
    #     loss = self.loss(outputs=outputs.getOutput('disp'), gts=gt, kitti=kitti, outMaxDisp=self.outMaxDisp)
    #     with self.ampHandle.scale_loss(loss.getLoss('Disp'), self.optimizer) as scaledLoss:
    #         scaledLoss.backward()
    #     self.optimizer.step()
    #
    #     outputs.addDisp(outputs.getOutput('disp')[2].detach(), self.outMaxDisp)
    #     return loss
    #
    # def train(self, batch: myUtils.Batch, kitti=False, weights=(), progress=0):
    #     pass
