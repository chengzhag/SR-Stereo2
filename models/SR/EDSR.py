import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
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
        rawOutput = super(RawEDSR, self).forward(imgL * self.args.rgb_range) / self.args.rgb_range
        output = {'outputSr': rawOutput}
        return output

    def load_state_dict(self, state_dict, strict=False):
        state_dict = myUtils.checkStateDict(
            model=self, stateDict=state_dict, strict=strict, possiblePrefix='sr.module')
        super().load_state_dict(state_dict, strict=False)


class EDSR(SR):
    def __init__(self, cInput=3, cuda=True, half=False):
        super().__init__(cInput=cInput, cuda=cuda, half=half)
        self.initModel()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001, betas=(0.9, 0.999))
        if self.cuda:
            self.model = nn.DataParallel(self.model)
            self.model.cuda()

    def initModel(self):
        self.model = RawEDSR(cInput=self.cInput)

    def packOutputs(self, outputs: dict, imgs: myUtils.Imgs = None) -> myUtils.Imgs:
        imgs = super().packOutputs(outputs, imgs)
        for key, value in outputs.items():
            if key.startswith('outputSr'):
                imgs.addImg(name=key, img=myUtils.quantize(value, 1))
        return imgs

    # outputs, gts: RGB value range 0~1
    def loss(self, output, gt):
        loss = myUtils.NameValues()
        # To get same loss with orignal EDSR, input range should scale to 0~self.args.rgb_range
        loss['lossSr'] = F.smooth_l1_loss(
            output['outputSr'] * self.model.module.args.rgb_range,
            gt * self.model.module.args.rgb_range,
            reduction='mean')
        loss['loss'] = loss['lossSr'] * self.lossWeights
        return loss

    def trainBothSides(self, inputs, gts):
        losses = myUtils.NameValues()
        outputs = myUtils.Imgs()
        for input, gt, side in zip(inputs, gts, ('L', 'R')):
            if gt is not None:
                loss, output = self.trainOneSide((input, ), gt)
                losses.update(nameValues=loss, suffix=side)
                outputs.update(imgs=output, suffix=side)

        return losses, outputs

    def train(self, batch: myUtils.Batch):
        batch.assertScales(2)
        return self.trainBothSides(batch.lowResRGBs(), batch.highResRGBs())
    
    def test(self, batch: myUtils.Batch, evalType: str):
        loss, outputs = super().test(batch=batch, evalType=evalType)
        for name in loss.keys():
            loss[name] *= self.model.module.args.rgb_range
        return loss, outputs




