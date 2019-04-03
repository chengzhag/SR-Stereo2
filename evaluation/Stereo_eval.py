import time
import torch
import os
from models import Stereo
from utils import myUtils
from evaluation.Evaluation import Evaluation as Base


class Evaluation(Base):
    def __init__(self, model, args, testImgLoader):
        stage, _ = os.path.splitext(os.path.basename(__file__))
        super().__init__(model=model, stage=stage, args=args, testImgLoader=testImgLoader)

    def _evalIt(self, batch: myUtils.Batch):
        loss, outputs = self.model.test(batch=batch,
                                        evalType=self.args.evalFcn,
                                        kitti=self.testImgLoader.kitti)
        for disp, input, side in zip(batch.lowestResDisps(), batch.lowestResRGBs(), ('L', 'R')):
            outputs.addImg('Disp', disp, range=self.model.outMaxDisp, prefix='gt', side=side)
            outputs.addImg('RGB', input, prefix='input', side=side)

        return loss, outputs

def main():
    import dataloader

    # Arguments
    args = myUtils.DefaultParser(description='evaluate Stereo net or SR-Stereo net')\
        .outputFolder().maxDisp().dispScale().model().dataPath()\
        .chkpoint().noCuda().seed().evalFcn().nSampleLog().dataset()\
        .loadScale().batchsize().half().resume().itRefine().validSetSample().parse()

    # Dataset
    _, testImgLoader = dataloader.getDataLoader(dataPath=args.dataPath,
                                                dataset=args.dataset,
                                                batchSizes=args.batchsize,
                                                loadScale=args.loadScale,
                                                mode='testing',
                                                validSetSample=args.validSetSample)

    stereo = getattr(Stereo, args.model)(
        maxDisp=args.maxDisp, dispScale=args.dispScale, cuda=args.cuda, half=args.half)
    if hasattr(stereo, 'itRefine'):
        stereo.itRefine = args.itRefine

    # Test
    test = Evaluation(model=stereo, args=args, testImgLoader=testImgLoader)
    test()


if __name__ == '__main__':
    main()
