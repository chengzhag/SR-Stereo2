import utils.experiment
import torch
import utils.data
import utils.imProcess
from evaluation import evalFcn
from utils import myUtils
from ..Model import Model


class Feature(Model):
    def __init__(self, cuda=True, half=False):
        super().__init__(cuda=cuda, half=half)
        self.cOutput = None

    def predict(self, batch: utils.data.Batch, mask=(1, 1)):
        self.model.eval()

        outputs = utils.imProcess.Imgs()
        with torch.no_grad():
            for input, do, side in zip(batch.lowestResRGBs(), mask, ('L', 'R')):
                if do:
                    output = self.packOutputs(self.model(input))
                    outputs.update(imgs=output, suffix=side)

        return outputs


