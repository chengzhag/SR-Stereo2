import torch.nn as nn
import torch.nn.functional as F
import utils.imProcess
from .SR import SR
from apex import amp


class RawBilinear(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        rawOutput = F.interpolate(input, scale_factor=2, mode='bilinear', align_corners=False)
        rawOutput = rawOutput.type_as(input)
        output = {'outputSr': rawOutput}
        return output

    def load_state_dict(self, state_dict, strict=False):
        pass

class Bilinear(SR):
    def __init__(self, cuda=True, half=False):
        super().__init__(cuda=cuda, half=half)
        self.initModel()
        self.optimizer = None
        if self.cuda:
            self.model.cuda()
            self.model = amp.initialize(models=self.model, enabled=half)
            self.model = nn.DataParallel(self.model)

    def initModel(self):
        self.model = RawBilinear()
        self.getParamNum()

    def packOutputs(self, outputs: dict, imgs: utils.imProcess.Imgs = None) -> utils.imProcess.Imgs:
        imgs = super().packOutputs(outputs, imgs)
        for key, value in outputs.items():
            if key.startswith('outputSr'):
                imgs.addImg(name=key, img=value)
        return imgs

