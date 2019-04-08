import torch
import torch.nn.functional as F
import torch.nn as nn
from utils import myUtils
from .RawPSMNet import stackhourglass as rawPSMNet
from .Stereo import Stereo
import torch.optim as optim


class RawPSMNetScale(rawPSMNet):
    def __init__(self, maxDisp, dispScale):
        super().__init__(maxDisp, dispScale)
        self.multiple = 16
        self.__imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                                 'std': [0.229, 0.224, 0.225]}

    # input: RGB value range 0~1
    # outputs: disparity range 0~self.maxdisp * self.dispScale
    def forward(self, left, right):
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
        output = {}
        output['outputDisp'] = rawOutputs
        return output

    def load_state_dict(self, state_dict, strict=False):
        myUtils.loadStateDict(model=self, stateDict=state_dict, strict=str)


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

    def packOutputs(self, outputs: dict, imgs: myUtils.Imgs = None) -> myUtils.Imgs:
        imgs = super().packOutputs(outputs, imgs)
        for key, value in outputs.items():
            if key == 'outputDisp':
                imgs.addImg(name=key, img=value, range=self.outMaxDisp)
        return imgs

    # input disparity maps:
    #   disparity range: 0~self.maxdisp * self.dispScale
    #   format: NCHW
    def loss(self, output: myUtils.Imgs, gt: torch.Tensor, outMaxDisp=None, kitti=False, weights=1):
        if outMaxDisp is None:
            outMaxDisp = self.outMaxDisp
        # for kitti dataset, only consider loss of none zero disparity pixels in gt
        mask = (gt > 0).detach() if kitti else (gt < outMaxDisp).detach()
        loss = myUtils.NameValues()
        loss['lossDisp'] = \
            0.5 * F.smooth_l1_loss(output['outputDisp'][0][mask], gt[mask], reduction='mean') \
            + 0.7 * F.smooth_l1_loss(output['outputDisp'][1][mask], gt[mask], reduction='mean') \
            + F.smooth_l1_loss(output['outputDisp'][2][mask], gt[mask], reduction='mean') \
            * weights
        loss['loss'] = loss['lossDisp']

        return loss

    def trainOneSide(self, input, gt, kitti=False):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.packOutputs(self.model.forward(*input))
        loss = self.loss(output=output,
                         gt=gt,
                         kitti=kitti,
                         weights=self.lossWeights)
        with self.ampHandle.scale_loss(loss['loss'], self.optimizer) as scaledLoss:
            scaledLoss.backward()
        self.optimizer.step()

        output.addImg(name='outputDisp', img=output['outputDisp'][2].detach(), range=self.outMaxDisp)
        return loss.dataItem(), output

    def trainBothSides(self, inputs, gts, kitti=False):
        losses = myUtils.NameValues()
        outputs = myUtils.Imgs()
        for input, gt, process, side in zip(
                inputs, gts,
                (lambda im: im, myUtils.flipLR),
                ('L', 'R')
        ):
            if gt is not None:
                loss, output = self.trainOneSide(
                    *process([input, gt]),
                    kitti=kitti
                )
                losses.update(nameValues=loss, suffix=side)
                outputs.update(imgs=process(output), suffix=side)

        return losses, outputs

    def train(self, batch: myUtils.Batch, kitti=False, progress=0):
        batch.assertScales(1)
        imgL, imgR = batch.highResRGBs()

        return self.trainBothSides(((imgL, imgR), (imgR, imgL)), batch.highResDisps(), kitti=kitti)
