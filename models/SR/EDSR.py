import utils.experiment
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import utils.data
import utils.imProcess
from utils import myUtils
from .RawEDSR import edsr
from .SR import SR
from apex import amp
from .RawEDSR import common
from ..Stereo.Feature import Feature


class RawEDSR(edsr.EDSR):
    def __init__(self, cInput):
        class Arg:
            def __init__(self):
                self.n_resblocks = 16
                self.n_feats = 64
                self.scale = [2]
                self.rgb_range = 255
                self.n_colors = 3
                self.n_inputs = cInput
                self.res_scale = 1
        self.args = Arg()
        super(RawEDSR, self).__init__(self.args)

    # input: RGB value range 0~1
    # output: RGB value range 0~1 without quantize
    def forward(self, imgL):
        rawOutput = super(RawEDSR, self).forward(imgL * self.args.rgb_range) / self.args.rgb_range
        output = {'outputSr': rawOutput}
        return output

    def load_state_dict(self, state_dict, strict=False):
        state_dict = utils.experiment.checkStateDict(
            model=self, stateDict=state_dict, strict=strict, possiblePrefix='sr.module.')
        super().load_state_dict(state_dict, strict=False)


class EDSR(SR):
    def __init__(self, cInput=3, cuda=True, half=False):
        super().__init__(cInput=cInput, cuda=cuda, half=half)
        self.initModel()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001, betas=(0.9, 0.999))
        if self.cuda:
            self.model.cuda()
            self.model, self.optimizer = amp.initialize(models=self.model, optimizers=self.optimizer, enabled=half)
            self.model = nn.DataParallel(self.model)

    def initModel(self):
        self.model = RawEDSR(cInput=self.cInput)
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
        for name in loss.keys():
            loss[name] *= self.model.module.args.rgb_range
        return loss

class RawEDSRfeature(nn.Module):

    def __init__(self):
        super().__init__()
        class Arg:
            def __init__(self):
                self.n_resblocks = 16
                self.n_feats = 64
                self.scale = [2]
                self.rgb_range = 255
                self.n_colors = 3
                self.n_inputs = 3
                self.res_scale = 1

        conv = common.default_conv
        args = Arg()
        self.args = args

        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale[0]
        act = nn.ReLU(True)
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        # define head module
        m_head = [conv(args.n_inputs, n_feats, kernel_size)]

        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)

    # input: RGB value range 0~1
    # output: Feature
    def forward(self, x):
        x = x * self.args.rgb_range

        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        output = {'outputFeature': res}

        return output

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

class EDSRfeature(Feature):
    def __init__(self, cuda=True, half=False):
        super().__init__(cuda=cuda, half=half)
        self.cOutput = 64
        self.initModel()
        if self.cuda:
            self.model.cuda()
            self.model = nn.DataParallel(self.model)

    def initModel(self):
        self.model = RawEDSRfeature()
        self.getParamNum()
