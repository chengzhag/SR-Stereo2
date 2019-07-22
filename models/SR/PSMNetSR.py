import utils.experiment
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import utils.data
import utils.imProcess
from utils import myUtils
from .SR import SR
from apex import amp
from .RawEDSR import common
from ..Stereo.PSMNet import RawPSMNetFeature

class RawPSMNetSR(nn.Module):
    def __init__(self):
        super().__init__()
        self.multiple = 4

        self.feature = RawPSMNetFeature()

        class Arg:
            def __init__(self):
                self.n_feats = 32
                self.rgb_range = 255
                self.n_colors = 3

        conv = common.default_conv
        args = Arg()
        self.args = args

        n_feats = args.n_feats
        kernel_size = 3
        scale = 2

        self.sub_mean = common.MeanShift(args.rgb_range)

        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            common.Upsampler(conv, scale, n_feats, act=False),
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)
        ]

        self.tail = nn.Sequential(*m_tail)

    # input: RGB value range 0~1
    # output: Feature
    def forward(self, x):
        x = x * self.args.rgb_range
        x = self.sub_mean(x)

        if self.training:
            x = self.feature(x)
            x = self.tail(x)
        else:
            autoPad = utils.imProcess.AutoPad(x, self.multiple, scale=2)
            x = autoPad.pad(x)
            x = self.feature(x)
            x = self.tail(x)
            x = autoPad.unpad(x)

        rawOutput = x / self.args.rgb_range
        output = {'outputSr': rawOutput}
        return output

    def load_state_dict(self, state_dict, strict=False):
        state_dict = utils.experiment.checkStateDict(
            model=self, stateDict=state_dict, strict=strict, possiblePrefix=('stereo.module.', 'module.stereoBody.'))
        super().load_state_dict(state_dict, strict=False)


class PSMNetSR(SR):
    def __init__(self, cuda=True, half=False):
        super().__init__(cuda=cuda, half=half)
        self.initModel()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001, betas=(0.9, 0.999))
        if self.cuda:
            self.model.cuda()
            self.model, self.optimizer = amp.initialize(models=self.model, optimizers=self.optimizer, enabled=half)
            self.model = nn.DataParallel(self.model)

    def initModel(self):
        self.model = RawPSMNetSR()
        self.getParamNum()

    def packOutputs(self, outputs: dict, imgs: utils.imProcess.Imgs = None) -> utils.imProcess.Imgs:
        imgs = super().packOutputs(outputs, imgs)
        for key, value in outputs.items():
            if key.startswith('outputSr'):
                imgs.addImg(name=key, img=utils.imProcess.quantize(value, 1))
        return imgs

    # outputs, gts: RGB value range 0~1
    def loss(self, output, gt):
        loss = utils.data.NameValues()
        # To get same loss with orignal EDSR, input range should scale to 0~self.args.rgb_range
        loss['lossSr'] = F.smooth_l1_loss(
            output['outputSr'] * self.model.module.args.rgb_range,
            gt * self.model.module.args.rgb_range,
            reduction='mean')
        loss['loss'] = loss['lossSr'] * self.lossWeights
        return loss

    def trainBothSides(self, inputs, gts):
        losses = utils.data.NameValues()
        outputs = utils.imProcess.Imgs()
        for input, gt, side in zip(inputs, gts, ('L', 'R')):
            if gt is not None:
                loss, output = self.trainOneSide((input, ), gt)
                losses.update(nameValues=loss, suffix=side)
                outputs.update(imgs=output, suffix=side)

        return losses, outputs

    def train(self, batch: utils.data.Batch):
        return self.trainBothSides(batch.lowResRGBs(), batch.highResRGBs())

    def testOutput(self, outputs: utils.imProcess.Imgs, gt, evalType: str):
        loss = super().testOutput(outputs=outputs, gt=gt, evalType=evalType)
        return loss