from .PSMNet import *


class RawStereoDown(nn.Module):
    def __init__(self, stereo):
        super().__init__()
        self.stereo = myUtils.getNNmoduleFromModel(stereo)
        self.pool = nn.AvgPool2d((2, 2))

    # input: RGB value range 0~1
    # outputs: disparity range 0~self.maxdisp * self.dispScale
    def forward(self, left, right):
        output = self.stereo.forward(left, right)
        output['outputDispHigh'] = output['outputDisp']
        output['outputDisp'] = myUtils.forNestingList(output['outputDisp'], lambda disp: self.pool(disp) / 2)
        return output

    def load_state_dict(self, state_dict, strict=False):
        state_dict = myUtils.checkStateDict(
            model=self, stateDict=state_dict, strict=str, possiblePrefix='stereo.module')
        super().load_state_dict(state_dict, strict=False)


class StereoDown(Stereo):
    def __init__(self, stereo):
        super().__init__(
            kitti=stereo.kitti, maxDisp=stereo.maxDisp, dispScale=stereo.dispScale, cuda=stereo.cuda, half=stereo.half)
        stereo.optimizer = None
        self.stereo = stereo
        self.initModel()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=0.001, betas=(0.9, 0.999)
        )
        if self.cuda:
            self.model = nn.DataParallel(self.model)
            self.model.cuda()

        self.outMaxDisp //= 2

    def setLossWeights(self, lossWeights):
        super().setLossWeights(lossWeights)
        self.stereo.setLossWeights(1)

    def initModel(self):
        self.model = RawStereoDown(self.stereo)

    def packOutputs(self, outputs: dict, imgs: myUtils.Imgs = None) -> myUtils.Imgs:
        imgs = super().packOutputs(outputs=outputs, imgs=imgs)
        imgs = self.stereo.packOutputs(outputs, imgs)
        imgs.range['outputDisp'] = self.outMaxDisp
        return imgs

    # input disparity maps:
    #   disparity range: 0~self.maxdisp * self.dispScale
    #   format: NCHW
    def loss(self, output: myUtils.Imgs, gt: tuple, outMaxDisp=None):
        if outMaxDisp is not None:
            raise Exception('Error: outputMaxDisp of PSMNetDown has no use!')
        loss = myUtils.NameValues()
        loss['loss'] = 0
        for name, g, outMaxDisp, weight in zip(
                ('DispHigh', 'Disp'),
                gt,
                (self.outMaxDisp * 2, self.outMaxDisp),
                self.lossWeights
        ):
            if g is not None:
                loss['loss' + name] = self.stereo.loss(
                    myUtils.Imgs(
                        (('outputDisp', output['output' + name]),)
                    ),
                    g, outMaxDisp=outMaxDisp
                )['lossDisp']
                loss['loss'] += weight * loss['loss' + name]
        return loss

    def train(self, batch: myUtils.Batch, progress=0):
        batch.assertScales(2)
        return self.trainBothSides(
            batch.highResRGBs(),
            list(zip(batch.highResDisps(), batch.lowResDisps())))

    def test(self, batch: myUtils.Batch, evalType: str):
        batch.assertScales(2)
        batch = myUtils.Batch(batch.highResRGBs() + batch.lowestResDisps(), cuda=batch.cuda, half=batch.half)

        return super(StereoDown, self).test(batch=batch, evalType=evalType)

    def load(self, checkpointDir):
        if checkpointDir is None:
            return None, None

        if (type(checkpointDir) in (list, tuple) and len(checkpointDir) == 1):
            return self.stereo.load(checkpointDir[0])
        elif type(checkpointDir) is str:
            try:
                # Complete load StereoDown from checkpoint
                return super().load(checkpointDir)
            except:
                # Only load stereo from checkpoint
                self.stereo.load(checkpointDir)
                return None, None
        raise Exception('Error: SRStereo need 2 checkpoints SR/Stereo or 1 checkpoint SRStereo to load!')
