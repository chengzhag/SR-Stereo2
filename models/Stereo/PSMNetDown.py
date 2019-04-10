from .PSMNet import *


class RawPSMNetDown(RawPSMNetScale):
    def __init__(self, maxDisp, dispScale):
        super().__init__(maxDisp=maxDisp, dispScale=dispScale)
        self.pool = nn.AvgPool2d((2, 2))

    # input: RGB value range 0~1
    # outputs: disparity range 0~self.maxdisp * self.dispScale
    def forward(self, left, right):
        output = super().forward(left, right)
        output['outputDispHigh'] = output['outputDisp']
        output['outputDisp'] = myUtils.forNestingList(output['outputDisp'], lambda disp: self.pool(disp) / 2)
        return output


class PSMNetDown(PSMNet):

    def __init__(self, maxDisp=192, dispScale=1, cuda=True, half=False):
        super().__init__(maxDisp=maxDisp, dispScale=dispScale, cuda=cuda, half=half)
        self.outMaxDisp //= 2

    def initModel(self):
        self.model = RawPSMNetDown(maxDisp=self.maxDisp, dispScale=self.dispScale)

    def packOutputs(self, outputs: dict, imgs: myUtils.Imgs = None) -> myUtils.Imgs:
        imgs = super().packOutputs(outputs=outputs, imgs=imgs)
        for key, value in outputs.items():
            if key.startswith('outputDispHigh'):
                imgs.addImg(name=key, img=value, range=self.outMaxDisp * 2)
        return imgs

    # input disparity maps:
    #   disparity range: 0~self.maxdisp * self.dispScale
    #   format: NCHW
    def loss(self, output: myUtils.Imgs, gt: tuple, outMaxDisp=None, kitti=False):
        if outMaxDisp is not None:
            raise Exception('Error: outputMaxDisp of PSMNetDown has no use!')
        loss = myUtils.NameValues()
        loss['loss'] = 0
        for name, g, outMaxDisp, weight in zip(
                ('DispHigh', 'Disp'),
                gt,
                (self.outMaxDisp * 2, self.maxDisp),
                self.lossWeights
        ):
            if g is not None:
                loss['loss' + name] = super().loss(
                    myUtils.Imgs(
                        (('outputDisp', output['output' + name]),)
                    ),
                    g, kitti=kitti, outMaxDisp=outMaxDisp
                )['lossDisp']
                loss['loss'] += weight * loss['loss' + name]
        return loss

    def trainOneSide(self, input, gt, kitti=False):
        loss, output = super().trainOneSide(input, gt, kitti=False)
        output.addImg(name='outputDispHigh', img=output['outputDispHigh'][2].detach(), range=self.maxDisp)
        return loss, output

    def train(self, batch: myUtils.Batch, kitti=False, progress=0):
        batch.assertScales(2)
        return self.trainBothSides(
            batch.highResRGBs(),
            list(zip(batch.highResDisps(), batch.lowResDisps())),
            kitti=kitti)

    def test(self, batch: myUtils.Batch, evalType: str, kitti=False):
        batch.assertScales(2)
        batch = myUtils.Batch(batch.highResRGBs() + batch.lowestResDisps(), cuda=batch.cuda, half=batch.half)

        return super(PSMNetDown, self).test(batch=batch, evalType=evalType, kitti=kitti)
