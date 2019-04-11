import torch
from evaluation import evalFcn
from utils import myUtils
from ..Model import Model


class SR(Model):
    def __init__(self, cInput=3, cuda=True, half=False):
        super().__init__(cuda=cuda, half=half)
        self.cInput = cInput

    def predict(self, batch: myUtils.Batch, mask=(1, 1)):
        batch.assertScales(1)
        self.model.eval()

        outputs = myUtils.Imgs()
        with torch.no_grad():
            for input, do, side in zip(batch.lowestResRGBs(), mask, ('L', 'R')):
                if do:
                    output = self.packOutputs(self.model(input))
                    outputs.update(imgs=output, suffix=side)

        return outputs

    def test(self, batch: myUtils.Batch, evalType: str):
        batch.assertScales(2)

        loss = myUtils.NameValues()
        mask = [gt is not None for gt in batch.highResRGBs()]
        outputs = self.predict(batch.lastScaleBatch(), mask)

        for gt, side in zip(batch.highResRGBs(), ('L', 'R')):
            output = outputs.get('outputSr' + side)
            if output is not None:
                loss[evalType + 'Sr' + side] = evalFcn.getEvalFcn(evalType)(gt, output)

        return loss, outputs

