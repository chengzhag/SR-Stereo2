import utils.experiment
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import utils.data
import utils.imProcess
from utils import myUtils
from ..Stereo.Feature import Feature
from apex import amp
from .RawEDSR import common

class RawMDSRfeature(nn.Module):

    def __init__(self):
        super().__init__()
        class Arg:
            def __init__(self):
                self.n_resblocks = 80
                self.n_feats = 64
                self.scale = [2, 3, 4]
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
        act = nn.ReLU(True)
        self.scale_idx = 0
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        self.pre_process = nn.ModuleList([
            nn.Sequential(
                common.ResBlock(conv, n_feats, 5, act=act),
                common.ResBlock(conv, n_feats, 5, act=act)
            ) for _ in args.scale
        ])

        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)

    def forward(self, x):
        x = x * self.args.rgb_range

        x = self.sub_mean(x)
        x = self.head(x)
        x = self.pre_process[self.scale_idx](x)

        res = self.body(x)
        res += x

        output = {'outputFeature': res}

        return output


class MDSRfeature(Feature):
    def __init__(self, cuda=True, half=False):
        super().__init__(cuda=cuda, half=half)
        self.cOutput = 64
        self.initModel()
        if self.cuda:
            self.model.cuda()
            self.model = nn.DataParallel(self.model)

    def initModel(self):
        self.model = RawMDSRfeature()
        self.getParamNum()

