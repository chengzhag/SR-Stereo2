from utils import myUtils
import os
from models import Stereo
from evaluation.Stereo_eval import Evaluation
from train.Train import Train as Base


class Train(Base):
    def __init__(self, test: Evaluation, trainImgLoader):
        super().__init__(test=test, trainImgLoader=trainImgLoader)

    def _trainIt(self, batch: myUtils.Batch):
        loss, outputs = self.experiment.model.train(batch=batch.detach(),
                                         kitti=self.trainImgLoader.kitti,)
        for disp, input, side in zip(batch.lowestResDisps(), batch.lowestResRGBs(), ('L', 'R')):
            outputs.addImg('Disp', disp, range=self.experiment.model.outMaxDisp, prefix='gt', side=side)
            outputs.addImg('Rgb', input, prefix='input', side=side)

        return loss, outputs.cpu()


def main():
    import dataloader

    # Arguments
    args = myUtils.DefaultParser(description='evaluate Stereo net or SR-Stereo net') \
        .outputFolder().maxDisp().dispScale().model().dataPath() \
        .chkpoint().noCuda().seed().evalFcn().nSampleLog().dataset() \
        .loadScale().batchSize().trainCrop().logEvery().testEvery() \
        .saveEvery().epochs().lr().lossWeights().subType() \
        .half().resume().itRefine().validSetSample().parse()

    # Dataset
    trainImgLoader, testImgLoader = dataloader.getDataLoader(dataPath=args.dataPath,
                                                 dataset=args.dataset,
                                                 trainCrop=args.trainCrop,
                                                 batchSizes=args.batchSize,
                                                 loadScale=args.loadScale,
                                                 mode='training' if args.subType is None else args.subType,
                                                 validSetSample=args.validSetSample)

    stereo = getattr(Stereo, args.model)(
        maxDisp=args.maxDisp, dispScale=args.dispScale, cuda=args.cuda, half=args.half)
    if hasattr(stereo, 'itRefine'):
        stereo.itRefine = args.itRefine

    # Train
    stage, _ = os.path.splitext(os.path.basename(__file__))
    experiment = myUtils.Experiment(model=stereo, stage=stage, args=args)
    test = Evaluation(experiment=experiment, testImgLoader=testImgLoader)
    train = Train(test=test, trainImgLoader=trainImgLoader)
    train()


if __name__ == '__main__':
    main()
