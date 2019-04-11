import torch
from evaluation import evalFcn
from utils import myUtils
from ..Model import Model


class Stereo(Model):
    def __init__(self, kitti, maxDisp=192, dispScale=1, cuda=True, half=False):
        super().__init__(cuda=cuda, half=half)
        self.kitti = kitti
        self.maxDisp = maxDisp
        self.dispScale = dispScale
        self.outMaxDisp = maxDisp * dispScale

    def loss(self, output, gt):
        pass

    def predict(self, batch: myUtils.Batch, mask=(1, 1)):
        batch.assertScales(1)
        self.model.eval()

        imgL, imgR = batch.lowestResRGBs()

        with torch.no_grad():
            outputs = myUtils.Imgs()
            for inputL, inputR, process, do, side in zip((imgL, imgR), (imgR, imgL),
                                                         (lambda im: im, myUtils.flipLR),
                                                         mask,
                                                         ('L', 'R')):
                if do:
                    output = process(self.packOutputs(self.model(process(inputL), process(inputR))))
                    outputs.update(imgs=output, suffix=side)

            return outputs

    def test(self, batch: myUtils.Batch, evalType: str):
        batch.assertScales(1)
        disps = batch.lowestResDisps()
        myUtils.assertDisp(*disps)

        loss = myUtils.NameValues()
        mask = [disp is not None for disp in disps]
        outputs = self.predict(batch, mask)

        for gt, side in zip(disps, ('L', 'R')):
            dispOut = outputs.get('outputDisp' + side)
            if dispOut is not None:
                # for kitti dataset, only consider loss of none zero disparity pixels in gt
                if self.kitti:
                    mask = gt > 0
                    dispOut = dispOut[mask]
                    gt = gt[mask]
                elif not self.kitti:
                    mask = gt < self.outMaxDisp
                    dispOut = dispOut[mask]
                    gt = gt[mask]
                loss[evalType + 'Disp' + side] = evalFcn.getEvalFcn(evalType)(gt, dispOut)

        return loss, outputs

    def trainBothSides(self, inputs, gts):
        losses = myUtils.NameValues()
        outputs = myUtils.Imgs()
        for input, gt, process, side in zip(
                (inputs, inputs[::-1]), gts,
                (lambda im: im, myUtils.flipLR),
                ('L', 'R')
        ):
            if (type(gt) not in (tuple, list) and gt is not None) \
                    or (gt is not None and not all([g is None for g in gt])):
                loss, output = self.trainOneSide(
                    *process([input, gt])
                )
                losses.update(nameValues=loss, suffix=side)
                outputs.update(imgs=process(output), suffix=side)

        return losses, outputs
