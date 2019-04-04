import torch
import torch.nn.functional as F
import torch.nn as nn
from utils import myUtils
from .RawPSMNet import stackhourglass as rawPSMNet
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

    def packOutputs(self, outputs: dict, imgs: myUtils.Imgs = None) -> myUtils.Imgs:
        if imgs is None:
            imgs = myUtils.Imgs()
        for key, value in outputs.items():
            if key == 'outputDisp':
                imgs.addImg(key, value, '', self.outMaxDisp)
        return imgs

    # input disparity maps:
    #   disparity range: 0~self.maxdisp * self.dispScale
    #   format: NCHW
    def loss(self, output: tuple, gts: torch.Tensor, outMaxDisp=None, kitti=False):
        if outMaxDisp is None:
            outMaxDisp = self.outMaxDisp
        # for kitti dataset, only consider loss of none zero disparity pixels in gt
        mask = (gts > 0).detach() if kitti else (gts < outMaxDisp).detach()
        loss = myUtils.Loss()
        loss.addLoss(
            0.5 * F.smooth_l1_loss(output[0][mask], gts[mask], reduction='mean')
            + 0.7 * F.smooth_l1_loss(output[1][mask], gts[mask], reduction='mean')
            + F.smooth_l1_loss(output[2][mask], gts[mask], reduction='mean'),
            name='Disp')

        return loss

    def trainOneSide(self, imgL, imgR, gt, kitti=False):
        self.optimizer.zero_grad()
        output = self.packOutputs(self.model.forward(imgL, imgR))
        loss = self.loss(output=output.getImg('Disp', prefix='output'),
                         gts=gt,
                         kitti=kitti,
                         outMaxDisp=self.outMaxDisp)
        with self.ampHandle.scale_loss(loss.getLoss('Disp'), self.optimizer) as scaledLoss:
            scaledLoss.backward()
        self.optimizer.step()

        output.addImg('Disp', output.getImg('Disp', prefix='output')[2].detach(), range=self.outMaxDisp,
                       prefix='output')
        return loss, output

    def train(self, batch: myUtils.Batch, kitti=False, weights=(), progress=0):
        batch.assertScales(1)
        super().train(batch=batch)

        imgL, imgR = batch.highResRGBs()

        losses = myUtils.Loss()
        outputs = myUtils.Imgs()
        for inputL, inputR, gt, process, side in zip(
                (imgL, imgR), (imgR, imgL),
                batch.highResDisps(),
                (lambda im: im, myUtils.flipLR),
                ('L', 'R')
        ):
            if gt is not None:
                loss, output = self.trainOneSide(
                    *process([inputL, inputR, gt]),
                    kitti=kitti
                )
                losses.update(nameValues=loss, suffix=side)
                outputs.update(imgs=process(output), suffix=side)


        return losses, outputs
