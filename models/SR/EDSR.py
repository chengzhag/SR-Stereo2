import torch.optim as optim
import torch.nn as nn
from utils import myUtils
from .RawEDSR import edsr
from .SR import SR


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
        rawOutput = super(RawEDSR, self).forward(imgL * self.rgb_range) / self.rgb_range
        if not self.training:
            rawOutput = myUtils.quantize(rawOutput, 1)
        output = {'outputSr': rawOutput}
        return output

    def load_state_dict(self, state_dict, strict=False):
        myUtils.loadStateDict(model=self, stateDict=state_dict, strict=str)


class EDSR(SR):
    def __init__(self, cuda=True, half=False):
        super().__init__(cuda=cuda, half=half)
        self.initModel()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001, betas=(0.9, 0.999))
        if self.cuda:
            self.model = nn.DataParallel(self.model)
            self.model.cuda()

    def initModel(self):
        self.model = RawEDSR(cInput=3)

    def packOutputs(self, outputs: dict, imgs: myUtils.Imgs = None) -> myUtils.Imgs:
        imgs = super().packOutputs(outputs, imgs)
        for key, value in outputs.items():
            if key == 'outputSr':
                imgs.addImg(name=key, img=value)
        return imgs



