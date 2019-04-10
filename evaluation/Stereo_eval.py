from utils import myUtils
import os
from models import Stereo
from evaluation.Evaluation import Evaluation as Base
import dataloader


class Evaluation(Base):

    def _evalIt(self, batch: myUtils.Batch):
        loss, outputs = self.experiment.model.test(batch=batch.detach(),
                                                   evalType=self.experiment.args.evalFcn)
        for disp, input, side in zip(batch.lowestResDisps(), batch.lowestResRGBs(), ('L', 'R')):
            outputs.addImg(name='gtDisp' + side, img=disp, range=self.experiment.model.outMaxDisp)
            outputs.addImg(name='inputRgb' + side, img=input)

        return loss, outputs.cpu()


def main():
    # Arguments
    args = myUtils.DefaultParser(description='evaluate Stereo net or SR-Stereo net') \
        .outputFolder().maxDisp().dispScale().model().dataPath() \
        .chkpoint().noCuda().seed().evalFcn().nSampleLog().dataset() \
        .loadScale().batchSize().half().resume().itRefine() \
        .validSetSample().noComet().subType().parse()

    # Dataset
    _, testImgLoader = dataloader.getDataLoader(dataPath=args.dataPath,
                                                dataset=args.dataset,
                                                batchSizes=args.batchSize,
                                                loadScale=args.loadScale,
                                                mode='testing' if args.subType is None else args.subType,
                                                validSetSample=args.validSetSample)

    # Model
    stereo = Stereo.getModel(
        args.model,
        kitti=testImgLoader.kitti,
        maxDisp=args.maxDisp,
        dispScale=args.dispScale,
        cuda=args.cuda,
        half=args.half)
    if hasattr(stereo, 'itRefine'):
        stereo.itRefine = args.itRefine

    # Test
    stage, _ = os.path.splitext(os.path.basename(__file__))
    experiment = myUtils.Experiment(model=stereo, stage=stage, args=args)
    test = Evaluation(experiment=experiment, testImgLoader=testImgLoader)
    test()


if __name__ == '__main__':
    main()
