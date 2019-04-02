import torch
from evaluation import evalFcn
from utils import myUtils
from ..Model import Model


class Stereo(Model):
    def __init__(self, maxDisp=192, dispScale=1, cuda=True, half=False):
        super().__init__(cuda=cuda, half=half)
        self.maxDisp = maxDisp
        self.dispScale = dispScale
        self.outMaxDisp = maxDisp * dispScale

    def predict(self, batch: myUtils.Batch, mask=(1, 1)):
        batch.assertScales(1)
        self.predict(batch)

        imgL, imgR = batch.lowestResRGBs()

        with torch.no_grad():
            outputs = myUtils.Output()
            for inputL, inputR, process, do, side in zip((imgL, imgR), (imgR, imgL),
                                                         (lambda im: im, myUtils.flipLR),
                                                         mask,
                                                         ('L', 'R')):
                if do:
                    output = process(self.model(process(inputL), process(inputR)))
                    outputs.update(output=output, suffix=side)

            return outputs

    def test(self, batch: myUtils.Batch, evalType: str, kitti=False):
        batch.assertScales(1)
        disps = batch.lowestResDisps()
        myUtils.assertDisp(*disps)

        loss = myUtils.Loss()
        mask = [disp is not None for disp in disps]
        outputs = self.predict(batch, mask)

        for gt, side in zip(disps, ('L', 'R')):
            dispOut = outputs.getOutput('disp')

            # for kitti dataset, only consider loss of none zero disparity pixels in gt
            if kitti:
                mask = gt > 0
                dispOut = dispOut[mask]
                gt = gt[mask]
            elif not kitti:
                mask = gt < self.outMaxDisp
                dispOut = dispOut[mask]
                gt = gt[mask]
            loss.addLoss(evalFcn.getEvalFcn(evalType)(gt, dispOut),
                         name='Disp', prefix=evalType, side=side)

        return loss, outputs

