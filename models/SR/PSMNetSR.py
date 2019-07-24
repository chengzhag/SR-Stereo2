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
# from ..Stereo.RawPSMNet.submodule import *
import torch
import torch.nn.functional as F
from ..Stereo.RawPSMNet.submodule import convbn, convbn_3d, BasicBlock

class feature_extraction(nn.Module):
    def __init__(self, cInput=3):
        super(feature_extraction, self).__init__()
        self.inplanes = 32
        self.firstconv = nn.Sequential(convbn(cInput, 32, 3, 2, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1,1,1)
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2,1,1)
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1,1,1)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1,1,2)

        self.branch1 = nn.Sequential(nn.AvgPool2d((64, 64), stride=(64,64)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch2 = nn.Sequential(nn.AvgPool2d((32, 32), stride=(32,32)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch3 = nn.Sequential(nn.AvgPool2d((16, 16), stride=(16,16)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch4 = nn.Sequential(nn.AvgPool2d((8, 8), stride=(8,8)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.lastconv = nn.Sequential(convbn(320, 128, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(128, 32, kernel_size=1, padding=0, stride = 1, bias=False))

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
           downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,1,None,pad,dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        output      = self.firstconv(x)
        inputBy2      = self.layer1(output)
        output_raw  = self.layer2(inputBy2)
        output      = self.layer3(output_raw)
        output_skip = self.layer4(output)


        output_branch1 = self.branch1(output_skip)
        output_branch1 = F.interpolate(output_branch1, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear',align_corners=False)

        output_branch2 = self.branch2(output_skip)
        output_branch2 = F.interpolate(output_branch2, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear',align_corners=False)

        output_branch3 = self.branch3(output_skip)
        output_branch3 = F.interpolate(output_branch3, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear',align_corners=False)

        output_branch4 = self.branch4(output_skip)
        output_branch4 = F.interpolate(output_branch4, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear',align_corners=False)

        output_feature = torch.cat((output_raw, output_skip, output_branch4, output_branch3, output_branch2, output_branch1), 1)
        output_feature = self.lastconv(output_feature)

        return output_feature, inputBy2

class RawPSMNetSR(nn.Module):
    def __init__(self):
        super().__init__()
        self.multiple = 4

        self.feature_extraction = feature_extraction(cInput=3)

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

        if not self.training:
            autoPad = utils.imProcess.AutoPad(x, self.multiple, scale=2)
            x = autoPad.pad(x)

        x, _ = self.feature_extraction(x)
        x = self.tail(x)

        if not self.training:
            x = autoPad.unpad(x)

        rawOutput = x / self.args.rgb_range
        output = {'outputSr': rawOutput}
        return output

    def load_state_dict(self, state_dict, strict=False):
        state_dict = utils.experiment.checkStateDict(
            model=self, stateDict=state_dict, strict=strict, possiblePrefix=('stereo.module.', 'module.stereoBody.'))
        super().load_state_dict(state_dict, strict=False)

class RawPSMNetSRfullHalfCat(nn.Module):
    def __init__(self):
        super().__init__()
        self.multiple = 4

        self.feature_extraction = feature_extraction(cInput=3)

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
        self.upsampler1 = common.Upsampler(conv, scale, n_feats, act=False)
        self.upsampler2 = common.Upsampler(conv, scale, n_feats * 2, act=False)
        self.upsampler3 = common.Upsampler(conv, scale, n_feats * 2 + args.n_colors, act=False)
        self.finalconv = conv(n_feats * 2 + args.n_colors, args.n_colors, kernel_size)


    # input: RGB value range 0~1
    # output: Feature
    def forward(self, input):
        input = input * self.args.rgb_range
        input = self.sub_mean(input)

        if not self.training:
            autoPad = utils.imProcess.AutoPad(input, self.multiple, scale=2)
            input = autoPad.pad(input)

        feature, inputBy2 = self.feature_extraction(input)
        outputBy4 = self.upsampler1(feature)
        cat1 = torch.cat((outputBy4,inputBy2), 1)
        outputBy2 = self.upsampler2(cat1)
        cat2 = torch.cat((outputBy2, input), 1)
        outputBy1 = self.upsampler3(cat2)
        outputBy1 = self.finalconv(outputBy1)

        if not self.training:
            outputBy1 = autoPad.unpad(outputBy1)

        rawOutput = outputBy1 / self.args.rgb_range
        output = {'outputSr': rawOutput}
        return output

    def load_state_dict(self, state_dict, strict=False):
        state_dict = utils.experiment.checkStateDict(
            model=self, stateDict=state_dict, strict=strict, possiblePrefix=('stereo.module.', 'module.stereoBody.'))
        super().load_state_dict(state_dict, strict=False)

def getPSMNetSR(rawPSMNetSR):
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
            self.model = rawPSMNetSR()
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
    return PSMNetSR



